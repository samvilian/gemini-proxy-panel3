const fetch = require('node-fetch');
const { Readable, Transform } = require('stream'); // Import Transform
const fsSync = require('fs'); // Synchronous fs for manual .env reading
const fs = require('fs').promises; // Async fs for temp file operations
const os = require('os');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const { GoogleGenAI } = require('@google/genai');
const configService = require('./configService');
const transformUtils = require('../utils/transform');

// List of Vertex AI supported models (prefix [v] indicates it's a Vertex API model)
const VERTEX_SUPPORTED_MODELS = [
    "[v]gemini-2.5-pro-exp-03-25",
    "[v]gemini-2.5-pro-preview-05-06",
    "[v]gemini-2.5-pro-preview-06-05",
    "[v]gemini-2.0-flash-001",
    "[v]gemini-2.5-flash-preview-04-17",
    "[v]gemini-2.5-flash-preview-05-20"
];

// Default region
const DEFAULT_REGION = 'global';

// Temporary credentials file path
let tempCredentialsPath = null;

// --- Manual .env Loading ---
let VERTEX_JSON_STRING = null; // Store manually loaded value

/**
 * Manually reads and parses the VERTEX variable from a .env file, handling multi-line JSON.
 * @param {string} [envFilePath=".env"] - Path to the .env file.
 */
function loadVertexEnvManual(envFilePath = ".env") {
    try {
        // Check if file exists synchronously
        if (!fsSync.existsSync(envFilePath)) {
            console.info(`Manual .env loading: ${envFilePath} not found.`); 
            VERTEX_JSON_STRING = null;
            return;
        }

        const fileContent = fsSync.readFileSync(envFilePath, 'utf-8');
        const lines = fileContent.split('\n');

        let vertexJsonLines = [];
        let inVertexVar = false;

        for (const line of lines) {
            const strippedLine = line.trim();
            if (!strippedLine || strippedLine.startsWith('#')) {
                continue; // Skip empty lines and comments
            }

            if (strippedLine.startsWith('VERTEX=')) {
                inVertexVar = true;
                const jsonPart = strippedLine.substring('VERTEX='.length).trim();
                // Check if JSON starts and potentially ends on the same line
                if (jsonPart.startsWith('{') && jsonPart.endsWith('}')) {
                    vertexJsonLines.push(jsonPart);
                    inVertexVar = false; // Finished single line JSON
                } else if (jsonPart.startsWith('{')) {
                    // Start of multi-line JSON
                    vertexJsonLines.push(jsonPart);
                } else {
                    console.warn(`Manual .env loading: VERTEX variable line starts unexpectedly: ${strippedLine}`); // Keep warn log in English
                    inVertexVar = false; // Reset if format is unexpected
                }
            } else if (inVertexVar) {
                vertexJsonLines.push(strippedLine);
                // Simple check for end of JSON object
                if (strippedLine.endsWith('}')) {
                    inVertexVar = false;
                }
            }
        }

        if (vertexJsonLines.length > 0) {
            const fullJsonString = vertexJsonLines.join(''); // Join lines without adding spaces/newlines
            try {
                // Validate JSON before assigning
                JSON.parse(fullJsonString);
                VERTEX_JSON_STRING = fullJsonString;
                console.info("Manual .env loading: Successfully parsed multi-line VERTEX JSON."); 
            } catch (jsonError) {
                console.error(`Manual .env loading: Extracted 'VERTEX' content from ${envFilePath} is not valid JSON: ${jsonError}. Content preview: ${fullJsonString.substring(0, 100)}...`); // Keep error log in English
                VERTEX_JSON_STRING = null;
            }
        } else {
             console.info(`Manual .env loading: VERTEX variable not found in ${envFilePath}.`); 
             VERTEX_JSON_STRING = null;
        }
    } catch (error) {
        console.error(`Manual .env loading: Error parsing ${envFilePath}:`, error); // Keep error log in English
        VERTEX_JSON_STRING = null;
    }
}

// --- Initialize Credentials on Load ---
let isVertexInitialized = false;
let isUsingExpressMode = false; // Track if we're using Express Mode

/**
 * Initializes Vertex credentials (loads JSON, creates temp file, sets env var) on service start.
 */
async function initializeVertexCredentials() {
    if (isVertexInitialized) return; // Already initialized

    // First check if Express Mode API Key is set (priority)
    const expressApiKey = process.env.EXPRESS_API_KEY;
    if (expressApiKey && typeof expressApiKey === 'string' && expressApiKey.trim()) {
        console.info("Using Vertex AI Express Mode with API key");
        isUsingExpressMode = true;
        isVertexInitialized = true; // Mark as initialized
        return; // No need for service account credentials
    }

    // If Express Mode not available, proceed with service account credentials
    const vertexJsonFromEnv = process.env.VERTEX;
    let potentialJsonString = null;
    let loadedFrom = ''; // Track where the JSON came from

    if (vertexJsonFromEnv && typeof vertexJsonFromEnv === 'string' && vertexJsonFromEnv.trim()) {
        try {
            // Validate if it's JSON
            JSON.parse(vertexJsonFromEnv);
            potentialJsonString = vertexJsonFromEnv;
            loadedFrom = 'process.env';
            console.info("Using VERTEX credentials from process.env."); 
        } catch (jsonError) {
            console.warn(`Manual .env loading: VERTEX variable from process.env is not valid JSON: ${jsonError}. Falling back to .env file.`); // Keep warn log in English
            potentialJsonString = null; // Invalidate if parse fails
        }
    }

    // If not found or invalid in process.env, try loading manually from .env file
    if (!potentialJsonString) {
        loadVertexEnvManual(); // This function sets VERTEX_JSON_STRING internally
        potentialJsonString = VERTEX_JSON_STRING; // Get the result from the manual load
        if (potentialJsonString) {
            loadedFrom = '.env file';
        }
    } else {
         // If loaded successfully from process.env, assign it to the module-level variable
         VERTEX_JSON_STRING = potentialJsonString;
    }

    // Check if credentials were ultimately found from either source
    if (!VERTEX_JSON_STRING) {
        console.log("Vertex AI credentials not found or invalid in process.env or .env file, Vertex AI disabled."); // Keep log in English
        isVertexInitialized = true; // Mark as initialized (but disabled)
        return;
    }

    try {
        const createdPath = await createServiceAccountFile(VERTEX_JSON_STRING);
        if (!createdPath) {
            throw new Error("createServiceAccountFile returned null or undefined.");
        }
        tempCredentialsPath = createdPath;
        process.env.GOOGLE_APPLICATION_CREDENTIALS = tempCredentialsPath;
        console.log(`Vertex AI credentials created and set: GOOGLE_APPLICATION_CREDENTIALS=${tempCredentialsPath}`); // Keep consolidated log in English
        isVertexInitialized = true;
    } catch (error) {
        console.error("!!! FATAL: Failed to initialize Vertex AI credentials:", error); // Keep error log in English
        VERTEX_JSON_STRING = null; // Disable Vertex if init fails
        tempCredentialsPath = null;
        isVertexInitialized = true; // Mark as initialized (but failed)
    }
}

// Initialize credentials when the module loads
initializeVertexCredentials();


/**
 * Creates a temporary service account file for Vertex AI authentication.
 * @param {string} vertexJsonString - JSON string containing the service account credentials.
 * @returns {Promise<string|null>} The path to the temporary file, or null on failure.
 */
async function createServiceAccountFile(vertexJsonString) {
    try {
        const serviceAccountInfo = JSON.parse(vertexJsonString);

        // Basic validation
        const requiredKeys = ["type", "project_id", "private_key_id", "private_key", "client_email", "client_id"];
        if (!requiredKeys.every(key => key in serviceAccountInfo)) {
            console.error("Invalid JSON format for 'VERTEX' environment variable. Missing required keys."); // Keep error log in English
            return null;
        }
        if (serviceAccountInfo.type !== "service_account") {
            console.error("Invalid JSON format for 'VERTEX' environment variable. 'type' must be 'service_account'."); // Keep error log in English
            return null;
        }

        const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'vertexai-nodejs-'));
        const tempFilePath = path.join(tempDir, 'service-account.json');
        await fs.writeFile(tempFilePath, JSON.stringify(serviceAccountInfo, null, 2), 'utf-8');
        // console.info(`Successfully parsed 'VERTEX' JSON and created temporary credentials file: ${tempFilePath}`); // Removed log
        return tempFilePath;
    } catch (e) {
        console.error(`Failed to create service account file from 'VERTEX' environment variable JSON: ${e}`, e); // Keep error log in English
        return null;
    }
}

/**
 * Maps OpenAI roles to Vertex AI roles.
 * @param {string} openaiRole - The role from the OpenAI request ('system', 'user', 'assistant', 'tool').
 * @returns {string} The corresponding Vertex AI role ('user', 'model', 'function').
 */
function mapOpenaiRoleToVertex(openaiRole) {
    const roleMap = {
        system: 'user', // Treat system messages as user messages for compatibility
        user: 'user',
        assistant: 'model',
        tool: 'function' // Tool results map to 'function' role in Vertex
    };
    return roleMap[openaiRole.toLowerCase()] || 'user'; // Default to user
}

/**
 * Parses a data URI (e.g., for base64 encoded images).
 * @param {string} uri - The data URI string.
 * @returns {{mimeType: string, data: Buffer}|null} Parsed mime type and data buffer, or null if invalid.
 */
function parseImageDataUri(uri) {
    if (!uri || !uri.startsWith('data:')) {
        return null;
    }
    try {
        const commaIndex = uri.indexOf(',');
        if (commaIndex === -1) return null;

        const header = uri.substring(5, commaIndex); // Remove 'data:' prefix
        const encodedData = uri.substring(commaIndex + 1);
        const parts = header.split(';');
        const mimeType = parts[0];

        if (parts.includes('base64')) {
            const data = Buffer.from(encodedData, 'base64');
            return { mimeType, data };
        } else {
            // Handle other encodings (e.g., URL encoding)
            console.warn(`Unsupported data URI encoding (non-base64): ${parts.slice(1).join(';')}`); // Keep warn log in English
            return { mimeType, data: Buffer.from(decodeURIComponent(encodedData)) }; // Attempt URL decoding
        }
    } catch (e) {
        console.error(`Error parsing data URI: ${e}`, e); // Keep error log in English
        return null;
    }
}

/**
 * Asynchronously converts OpenAI message content parts to Vertex AI Parts, handling text and images.
 * Downloads images from HTTPS URLs if necessary.
 * @param {Array<object>} openAIContentParts - Array of OpenAI content parts (text or image_url).
 * @returns {Promise<Array<object>>} A promise resolving to an array of Vertex AI Part objects.
 */
async function convertOpenaiPartsToVertexParts(openAIContentParts) {
    const vertexParts = [];
    for (const part of openAIContentParts) {
        if (part.type === 'text') {
            vertexParts.push({ text: part.text });
        } else if (part.type === 'image_url' && part.image_url) {
            const imageUrl = part.image_url.url;
            if (imageUrl.startsWith('data:')) {
                const parsed = parseImageDataUri(imageUrl);
                if (parsed) {
                    vertexParts.push({
                        inlineData: {
                            mimeType: parsed.mimeType,
                            data: parsed.data.toString('base64') // Vertex SDK expects base64 string
                        }
                    });
                } else {
                    console.warn(`Could not parse data URI: ${imageUrl.substring(0, 50)}...`); // Keep warn log in English
                    vertexParts.push({ text: `[Failed to parse image data URI]` });
                }
            } else if (imageUrl.startsWith('gs://')) {
                // Handle Google Cloud Storage URIs
                const mime = require('mime-types'); // Lazy require mime-types
                vertexParts.push({
                    fileData: {
                        mimeType: mime.lookup(imageUrl) || 'application/octet-stream',
                        fileUri: imageUrl
                    }
                });
            } else if (imageUrl.startsWith('http://') || imageUrl.startsWith('https://')) {
                // Attempt to download image from URL
                try {
                    const response = await fetch(imageUrl, { timeout: 10000 }); // 10s timeout
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const imageBuffer = await response.buffer();
                    const contentType = response.headers.get('content-type') || 'application/octet-stream';
                    vertexParts.push({
                        inlineData: {
                            mimeType: contentType,
                            data: imageBuffer.toString('base64')
                        }
                    });
                } catch (e) {
                    console.error(`Failed to download image from ${imageUrl}: ${e}`); // Keep error log in English
                    vertexParts.push({ text: `[Failed to load image at ${imageUrl}]` });
                }
            } else {
                console.warn(`Unsupported image URL format: ${imageUrl}`); // Keep warn log in English
                vertexParts.push({ text: `[Unsupported image format at ${imageUrl}]` });
            }
        }
    }
    return vertexParts;
}

/**
 * Converts OpenAI message list to Vertex AI Content array.
 * @param {Array<object>} messages - Array of OpenAI message objects.
 * @returns {Promise<Array<object>>} A promise resolving to an array of Vertex AI Content objects.
 */
async function convertOpenaiMessagesToVertex(messages) {
    const vertexContents = [];
    
    // Process all messages, including system messages, mapping to appropriate Vertex roles
    for (const msg of messages) {
        const vertexRole = mapOpenaiRoleToVertex(msg.role);
        let parts = [];

        if (vertexRole === 'function') { // Handle tool/function results
            if (msg.tool_call_id && msg.content) {
                if (msg.name) {
                    let responseContent = {};
                    try {
                        // Attempt to parse the string content into an object
                        responseContent = JSON.parse(msg.content);
                    } catch (e) {
                        console.warn(`Tool result content for ${msg.name} (${msg.tool_call_id}) is not valid JSON, sending as string: ${msg.content}`); // Keep warn log in English
                        // Send as simple text if not parsable JSON
                        parts.push({ text: `[Tool Result for ${msg.name}: ${msg.content}]` });
                        continue; // Skip adding as functionResponse if invalid
                    }
                    parts.push({
                        functionResponse: {
                            name: msg.name,
                            response: responseContent // Vertex SDK expects the actual object
                        }
                    });
                } else {
                    console.warn(`Tool message received without function name (expected in msg.name): ${JSON.stringify(msg)}`); // Keep warn log in English
                    parts.push({ text: `[Tool Result for ${msg.tool_call_id}: ${msg.content}]` });
                }
            } else {
                console.warn(`Tool message missing tool_call_id or content: ${JSON.stringify(msg)}`); // Keep warn log in English
                parts.push({ text: msg.content || '[Empty Tool Message]' });
            }
        } else if (vertexRole === 'model') { // Handle assistant messages (including potential tool calls)
            if (msg.tool_calls && msg.tool_calls.length > 0) {
                // If assistant message contains tool calls, represent them as FunctionCallParts
                for (const toolCall of msg.tool_calls) {
                    if (toolCall.type === 'function' && toolCall.function) {
                        let args = {};
                        try {
                            // Arguments from OpenAI are a JSON string, Vertex expects an object
                            args = JSON.parse(toolCall.function.arguments || '{}');
                        } catch (e) {
                            console.error(`Failed to parse tool call arguments for ${toolCall.function.name}: ${e}`); // Keep error log in English
                            args = { _error: "Failed to parse arguments", raw_arguments: toolCall.function.arguments };
                        }
                        parts.push({
                            functionCall: {
                                name: toolCall.function.name,
                                args: args // Pass the parsed object
                            }
                        });
                    }
                }
                // If there's also text content along with tool calls, add it as a separate text part
                if (msg.content && typeof msg.content === 'string') {
                    parts.push({ text: msg.content });
                } else if (Array.isArray(msg.content)) {
                    // Handle multi-part assistant messages (rare but possible)
                    const textParts = msg.content.filter(p => p.type === 'text').map(p => p.text).join('\n');
                    if (textParts) {
                        parts.push({ text: textParts });
                    }
                    // Note: Image parts from assistant messages are generally not expected/handled here.
                }
            } else {
                // Normal assistant message (text or potentially multimodal)
                if (typeof msg.content === 'string') {
                    parts.push({ text: msg.content });
                } else if (Array.isArray(msg.content)) {
                    parts = parts.concat(await convertOpenaiPartsToVertexParts(msg.content));
                }
            }
        } else { // Handle 'user' messages (can be text or multimodal)
            if (typeof msg.content === 'string') {
                parts.push({ text: msg.content });
            } else if (Array.isArray(msg.content)) {
                parts = parts.concat(await convertOpenaiPartsToVertexParts(msg.content));
            }
        }

        if (parts.length > 0) {
            // Ensure role mapping is correct before pushing
            const finalVertexRole = mapOpenaiRoleToVertex(msg.role);
            vertexContents.push({ role: finalVertexRole, parts });
        } else {
            console.warn(`Message resulted in empty parts, skipping: ${JSON.stringify(msg)}`); // Keep warn log in English
        }
    }

    return vertexContents;
}

/**
 * Converts OpenAI tool definitions to Vertex AI Tool format.
 * @param {Array<object>|null} tools - Array of OpenAI tool objects.
 * @returns {Array<object>|null} Array of Vertex AI Tool objects or null.
 */
function convertOpenaiToolsToVertex(tools) {
    if (!tools || tools.length === 0) {
        return null;
    }

    const functionDeclarations = [];
    for (const tool of tools) {
        if (tool.type === 'function' && tool.function) {
            const func = tool.function;
            functionDeclarations.push({
                name: func.name,
                description: func.description || '',
                // Pass the parameters object directly, assuming it's compatible enough for the SDK
                parameters: func.parameters || { type: 'object', properties: {} } // Provide default empty schema if none
            });
        } else {
            console.warn(`Unsupported tool type encountered: ${tool.type}`); // Keep warn log in English
        }
    }

    if (functionDeclarations.length > 0) {
        // Vertex SDK expects a Tool object containing the declarations
        return [{ functionDeclarations }];
    }

    return null;
}

/**
 * Maps Vertex AI finish reasons to OpenAI finish reasons.
 * @param {string|null} reason - Vertex AI finish reason string.
 * @returns {string|null} OpenAI finish reason string or null.
 */
function convertVertexFinishReasonToOpenai(reason) {
    if (!reason) return null;
    const mapping = {
        'STOP': 'stop',
        'MAX_TOKENS': 'length',
        'SAFETY': 'content_filter',
        'RECITATION': 'content_filter', // Often related to safety/policy
        'TOOL_CALL': 'tool_calls',
        'FUNCTION_CALL': 'tool_calls', // Older naming
        'FINISH_REASON_UNSPECIFIED': null,
        'OTHER': null
    };
    return mapping[reason.toUpperCase()] || null; // Default to null if unknown
}

/**
 * Converts a Vertex FunctionCallPart or FunctionCall object into an OpenAI tool_calls object.
 * @param {object} functionCall - The Vertex functionCall object.
 * @param {number} [index=0] - Optional index for multiple tool calls.
 * @returns {object|null} OpenAI tool_calls object structure, or null if input is invalid.
 */
function convertVertexToolCallToOpenai(functionCall, index = 0) {
    if (!functionCall || !functionCall.name) {
        console.error("Invalid functionCall object received from Vertex", functionCall); // Keep error log in English
        return null;
    }
    return {
        id: `call_${uuidv4()}`, // Generate a unique ID for the call
        type: 'function',
        function: {
            name: functionCall.name,
            // OpenAI expects arguments as a JSON string
            arguments: JSON.stringify(functionCall.args || {})
        },
        index: index
    };
}

/**
 * Creates Vertex AI safety settings.
 * @param {string} [blockLevel='OFF'] - The threshold level.
 * @returns {Array<object>} Array of Vertex safety setting objects.
 */
function createSafetySettings(blockLevel = 'OFF') {
    const HarmCategory = {
        HARM_CATEGORY_UNSPECIFIED: "HARM_CATEGORY_UNSPECIFIED",
        HARM_CATEGORY_HATE_SPEECH: "HARM_CATEGORY_HATE_SPEECH",
        HARM_CATEGORY_DANGEROUS_CONTENT: "HARM_CATEGORY_DANGEROUS_CONTENT",
        HARM_CATEGORY_HARASSMENT: "HARM_CATEGORY_HARASSMENT",
        HARM_CATEGORY_SEXUALLY_EXPLICIT: "HARM_CATEGORY_SEXUALLY_EXPLICIT"
    };

    const categories = [
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT
    ];
    return categories.map(category => ({
        category: category,
        threshold: blockLevel // Use the string representation
    }));
}

/**
 * Handles chat completion requests for the Vertex API.
 */
async function proxyVertexChatCompletions(openAIRequestBody, workerApiKey, stream) {
    console.log("Using Vertex AI proxy service"); // Keep log in English
    
    // Whether to use KEEPALIVE in streaming mode
    const keepAliveEnabled = process.env.KEEPALIVE === '1';
    const requestedModelId = openAIRequestBody?.model;

    // Validate request
    if (!requestedModelId) {
        return { error: { message: "Missing 'model' field in request body" }, status: 400 };
    }
    if (!openAIRequestBody.messages || !Array.isArray(openAIRequestBody.messages)) {
        return { error: { message: "Missing or invalid 'messages' field in request body" }, status: 400 };
    }

    // Remove [v] prefix from model name to get the actual Vertex model ID
    let vertexModelId = requestedModelId;
    if (vertexModelId.startsWith('[v]')) {
        vertexModelId = vertexModelId.substring(3);
    }

    // Check safety setting
    let isSafetyEnabled;
    try {
        isSafetyEnabled = await configService.getWorkerKeySafetySetting(workerApiKey);
    } catch (error) {
        console.error("Error getting worker key safety setting:", error); // Keep error log in English
        isSafetyEnabled = true; // Default to enabled safety settings
    }

    // Check initialization status
    if (!isVertexInitialized) {
        return {
            error: {
                message: "Vertex AI service not initialized."
            },
            status: 500
        };
    }

    let ai;
    
    try {
        // Initialize client based on authentication mode
        if (isUsingExpressMode) {
            // Express Mode with API Key
            const expressApiKey = process.env.EXPRESS_API_KEY;
            if (!expressApiKey) {
                throw new Error("EXPRESS_API_KEY is not available.");
            }
            
            ai = new GoogleGenAI({
                vertexai: true,
                apiKey: expressApiKey
            });
            
            console.log("Vertex AI Client initialized with Express Mode API Key"); // Keep log in English
        } else {
            // Standard mode with service account
            if (!tempCredentialsPath) {
                return {
                    error: {
                        message: "Temporary credentials path not set for service account authentication."
                    },
                    status: 500
                };
            }
            
            // Read service account file to get project_id
            const keyFileContent = await fs.readFile(tempCredentialsPath, 'utf-8');
            const keyFileData = JSON.parse(keyFileContent);
            const project_id = keyFileData.project_id;
            
            if (!project_id) {
                throw new Error("No project_id found in service account JSON");
            }

            // Initialize GoogleGenAI client with Vertex AI service account configuration
            let region = DEFAULT_REGION;
            ai = new GoogleGenAI({
                vertexai: true,
                project: project_id,
                location: region
            });
            
            console.log(`Vertex AI Client initialized for project '${project_id}' in region '${region}'.`); // Keep log in English
        }

        // Convert OpenAI format to Vertex format
        const vertexContents = await convertOpenaiMessagesToVertex(openAIRequestBody.messages);
        const vertexTools = convertOpenaiToolsToVertex(openAIRequestBody.tools);
        
        // Set safety level
        const safetySettings = createSafetySettings(isSafetyEnabled ? 'BLOCK_MEDIUM_AND_ABOVE' : 'OFF');

        // Configure generation parameters
        const generationConfig = {
            maxOutputTokens: openAIRequestBody.max_tokens,
            temperature: openAIRequestBody.temperature,
            topP: openAIRequestBody.top_p,
            topK: openAIRequestBody.top_k,
            stopSequences: typeof openAIRequestBody.stop === 'string' ? [openAIRequestBody.stop] : openAIRequestBody.stop
        };
        
        // Remove undefined keys
        Object.keys(generationConfig).forEach(key => 
            generationConfig[key] === undefined && delete generationConfig[key]
        );

        // Tool configuration
        let toolConfig = null;
        if (vertexTools) {
            let mode = 'AUTO'; // Default
            let allowedFunctionNames = [];

            if (openAIRequestBody.tool_choice) {
                if (typeof openAIRequestBody.tool_choice === 'string') {
                    if (openAIRequestBody.tool_choice === 'none') {
                        mode = 'NONE';
                    } else if (openAIRequestBody.tool_choice === 'auto') {
                        mode = 'AUTO';
                    } else {
                        mode = 'ANY'; // Assume non-standard string is a function name
                        allowedFunctionNames.push(openAIRequestBody.tool_choice);
                    }
                } else if (typeof openAIRequestBody.tool_choice === 'object' && openAIRequestBody.tool_choice.type === 'function') {
                    const funcName = openAIRequestBody.tool_choice.function?.name;
                    if (funcName) {
                        mode = 'ANY'; // ANY requires specifying function name(s)
                        allowedFunctionNames.push(funcName);
                    }
                }
            }
            toolConfig = {
                functionCallingConfig: {
                    mode: mode,
                    allowedFunctionNames: allowedFunctionNames.length > 0 ? allowedFunctionNames : undefined
                }
            };
        }

        // Build the request payload with all parameters
        const requestPayload = {
            model: vertexModelId,
            contents: vertexContents,
            generationConfig: generationConfig,
            safetySettings: safetySettings,
            tools: vertexTools,
            toolConfig: toolConfig
        };
        // Remove keys with null or undefined values from the payload
        Object.keys(requestPayload).forEach(key => (requestPayload[key] == null) && delete requestPayload[key]);


        // Determine if KEEPALIVE mode should be used:
        // 1. KEEPALIVE environment variable is set to 1
        // 2. Client requested streaming
        // 3. Safety settings are disabled
        const useKeepAlive = keepAliveEnabled && stream && !isSafetyEnabled;
        
        // Determine the actual stream mode based on whether KEEPALIVE is used
        const actualStreamMode = useKeepAlive ? false : stream;
        
        // Log KEEPALIVE mode
        if (useKeepAlive) {
            console.log(`Using KEEPALIVE mode: Client requested streaming but sending non-streaming request to Vertex (safety settings disabled)`); // Keep log in English
        }

        // Handle response
        if (stream) {
            if (useKeepAlive) {
                // KEEPALIVE mode: Use non-streaming API but respond to client in a streaming way
                try {
                    // Send non-streaming request to Vertex using the new API
                    const response = await ai.models.generateContent(requestPayload);
                    
                    if (!response || !response.candidates || response.candidates.length === 0) {
                        // Check if blocked by safety filter
                        const promptFeedback = response?.promptFeedback;
                        if (promptFeedback?.blockReason) {
                            const blockMessage = promptFeedback.blockReasonMessage || `Blocked due to ${promptFeedback.blockReason}`;
                            console.warn(`Request blocked by safety filters: ${blockMessage}`); // Keep warn log in English
                            return {
                                error: {
                                    message: `Request blocked by Vertex AI safety filters: ${blockMessage}`,
                                    type: "vertex_ai_safety_filter",
                                    code: "content_filter"
                                },
                                status: 400
                            };
                        }
                        throw new Error("No valid candidates received from Vertex AI.");
                    }
                    
                    console.log(`Completed KEEPALIVE mode chat request`); // Keep log in English
                    
                    // Return an object containing a promise for the response
                    // This allows apiV1.js to start sending keep-alives immediately
                    // while awaiting the actual Vertex response.
                    const getResponsePromise = async () => {
                        try {
                            const response = await ai.models.generateContent(requestPayload);
                            if (!response || !response.candidates || response.candidates.length === 0) {
                                const promptFeedback = response?.promptFeedback;
                                if (promptFeedback?.blockReason) {
                                    const blockMessage = promptFeedback.blockReasonMessage || `Blocked due to ${promptFeedback.blockReason}`;
                                    console.warn(`Request blocked by safety filters: ${blockMessage}`);
                                    // Propagate a specific error structure
                                    const error = new Error(`Request blocked by Vertex AI safety filters: ${blockMessage}`);
                                    error.type = "vertex_ai_safety_filter";
                                    error.code = "content_filter";
                                    error.status = 400;
                                    throw error;
                                }
                                throw new Error("No valid candidates received from Vertex AI.");
                            }
                            console.log(`Completed KEEPALIVE mode chat request to Vertex`);
                            return response; // Return the actual Vertex response
                        } catch (error) {
                            console.error(`Error during Vertex AI KEEPALIVE generation (inside promise): ${error}`, error);
                            // Re-throw or wrap error to be caught by apiV1.js
                            const wrappedError = new Error(`Vertex AI KEEPALIVE generation failed: ${error.message}`);
                            wrappedError.type = error.type || 'vertex_ai_error';
                            wrappedError.status = error.status || 500;
                            wrappedError.originalError = error;
                            throw wrappedError;
                        }
                    };
                    
                    return {
                        getResponsePromise: getResponsePromise(), // Execute and return the promise
                        selectedKeyId: 'vertex-ai',
                        modelCategory: 'Vertex',
                        isKeepAlive: true,
                        requestedModelId: requestedModelId
                    };

                } catch (error) {
                    // This catch block might be for errors *before* calling ai.models.generateContent
                    // or if the promise construction itself fails.
                    console.error(`Error setting up Vertex AI KEEPALIVE generation: ${error}`, error);
                    return {
                        error: {
                            message: `Vertex AI KEEPALIVE setup failed: ${error.message}`,
                            type: 'vertex_ai_setup_error'
                        },
                        status: 500
                    };
                }
            } else {
                // Standard streaming mode
                try {
                    // Use the new API for streaming
                    const streamResult = await ai.models.generateContentStream(requestPayload);
                    
                    let toolCallIndex = 0; // Keep track across chunks

                    // Create a Transform stream to process the stream from Vertex SDK
                    const vertexTransformer = new Transform({
                        objectMode: true, // Process objects from Vertex SDK
                        async transform(item, encoding, callback) {
                            try {
                                if (!item || !item.candidates || item.candidates.length === 0) {
                                    return callback(); // Skip empty items
                                }

                                const candidate = item.candidates[0];
                                const finishReasonVertex = candidate?.finishReason;
                                const finishReasonOpenai = convertVertexFinishReasonToOpenai(finishReasonVertex);

                                let deltaContent = null;
                                let deltaToolCalls = [];

                                if (candidate.content && candidate.content.parts) {
                                    for (const part of candidate.content.parts) {
                                        if (part.text) {
                                            deltaContent = part.text;
                                        } else if (part.functionCall) {
                                            const openaiToolCall = convertVertexToolCallToOpenai(part.functionCall, toolCallIndex++);
                                            if (openaiToolCall) {
                                                deltaToolCalls.push(openaiToolCall);
                                            }
                                        }
                                    }
                                }

                                // Create chunk only if there's content, tool calls, or a finish reason
                                if (deltaContent !== null || deltaToolCalls.length > 0 || finishReasonOpenai) {
                                    const choiceDelta = {
                                        role: 'assistant',
                                        content: deltaContent,
                                        tool_calls: deltaToolCalls.length > 0 ? deltaToolCalls : undefined
                                    };
                                    const streamChoice = {
                                        index: 0,
                                        delta: choiceDelta,
                                        finish_reason: finishReasonOpenai,
                                        logprobs: null
                                    };
                                    const responseChunk = {
                                        id: `chatcmpl-stream-${uuidv4()}`,
                                        object: 'chat.completion.chunk',
                                        created: Math.floor(Date.now() / 1000),
                                        model: requestedModelId,
                                        choices: [streamChoice],
                                        usage: null
                                    };
                                    // Push the transformed JSON string downstream
                                    this.push(JSON.stringify(responseChunk));
                                }
                                
                                // Prepare to end if there's a finish reason
                                // Note: No need to explicitly end the stream here, let the source stream end naturally
                                callback();
                            } catch (err) {
                                callback(err); // Propagate errors
                            }
                        },

                        flush(callback) {
                            // Getting final aggregated usage data is difficult here as we are a transform stream
                            // Ignore sending aggregated data for now
                            
                            // Send the [DONE] message
                            this.push(JSON.stringify({ done: true }));
                            callback();
                        }
                    });

                    // Pipe the Vertex SDK stream through our transformer
                    // The streamResult might be structured differently based on the API mode
                    // In standard mode: streamResult.stream is AsyncIterable<GenerateContentResponse>
                    // In Express Mode: streamResult itself might be the iterable
                    let sdkStream;
                    
                    if (streamResult.stream) {
                        // Standard mode structure with .stream property
                        sdkStream = Readable.from(streamResult.stream);
                    } else if (streamResult[Symbol.asyncIterator] || streamResult[Symbol.iterator]) {
                        // Express Mode might return the iterator directly
                        sdkStream = Readable.from(streamResult);
                    } else {
                        throw new Error("Unexpected response format from Vertex AI streaming API");
                    }
                    
                    const outputStream = sdkStream.pipe(vertexTransformer);

                    return {
                        response: { body: outputStream }, // Return the transform stream directly
                        selectedKeyId: 'vertex-ai',
                        modelCategory: 'Vertex'
                    };

                } catch (error) {
                    console.error(`Error during Vertex AI stream generation: ${error}`, error); // Keep error log in English
                    return {
                        error: {
                            message: `Vertex AI stream generation failed: ${error.message}`,
                            type: 'vertex_ai_error'
                        },
                        status: 500
                    };
                }
            }
        } else {
            // Non-streaming response
            try {
                // Use the new API for non-streaming
                const response = await ai.models.generateContent(requestPayload);

                if (!response || !response.candidates || response.candidates.length === 0) {
                    // Check if blocked by safety filter
                    const promptFeedback = response?.promptFeedback;
                    if (promptFeedback?.blockReason) {
                        const blockMessage = promptFeedback.blockReasonMessage || `Blocked due to ${promptFeedback.blockReason}`;
                        console.warn(`Request blocked by safety filters: ${blockMessage}`); // Keep warn log in English
                        return {
                            error: {
                                message: `Request blocked by Vertex AI safety filters: ${blockMessage}`,
                                type: "vertex_ai_safety_filter",
                                code: "content_filter"
                            },
                            status: 400
                        };
                    }
                    throw new Error("No valid candidates received from Vertex AI.");
                }

                const candidate = response.candidates[0];
                const finishReasonVertex = candidate.finishReason;
                const finishReasonOpenai = convertVertexFinishReasonToOpenai(finishReasonVertex);

                let responseContent = null;
                let responseToolCalls = [];

                if (candidate.content && candidate.content.parts) {
                    const textParts = [];
                    for (const part of candidate.content.parts) {
                        if (part.text) {
                            textParts.push(part.text);
                        } else if (part.functionCall) {
                            const openaiToolCall = convertVertexToolCallToOpenai(part.functionCall);
                            if (openaiToolCall) {
                                responseToolCalls.push(openaiToolCall);
                            }
                        }
                    }
                    if (textParts.length > 0) {
                        responseContent = textParts.join(''); // Concatenate text parts
                    }
                }

                // Handle response blocked by safety filter
                if (finishReasonOpenai === 'content_filter' && !responseContent && responseToolCalls.length === 0) {
                    const safetyRatings = candidate.safetyRatings || [];
                    const blockMessages = safetyRatings.filter(r => r.blocked).map(r => `${r.category}: ${r.severity || 'Blocked'}`);
                    const message = `Response blocked by Vertex AI safety filters. Reasons: ${blockMessages.join(' ') || finishReasonVertex}`;
                    console.warn(message); // Keep warn log in English
                    return {
                        error: {
                            message: message,
                            type: "vertex_ai_safety_filter",
                            code: "content_filter"
                        },
                        status: 400
                    };
                }

                // Build OpenAI format response message
                const message = {
                    role: 'assistant',
                    content: responseContent, // Can be null if only tool calls
                    tool_calls: responseToolCalls.length > 0 ? responseToolCalls : undefined
                };

                const choice = {
                    index: 0,
                    message: message,
                    finish_reason: finishReasonOpenai,
                    logprobs: null // Not supported
                };

                // Extract usage statistics
                const usage = {
                    prompt_tokens: response.usageMetadata?.promptTokenCount || 0,
                    completion_tokens: response.usageMetadata?.candidatesTokenCount || 0,
                    total_tokens: response.usageMetadata?.totalTokenCount || (response.usageMetadata?.promptTokenCount || 0) + (response.usageMetadata?.candidatesTokenCount || 0) // Calculate if not present
                };

                // Create the full OpenAI format response
                const openaiResponse = {
                    id: `chatcmpl-${uuidv4()}`,
                    object: 'chat.completion',
                    created: Math.floor(Date.now() / 1000),
                    model: requestedModelId,
                    choices: [choice],
                    usage: usage,
                    system_fingerprint: null // Not provided by Vertex
                };

                // Format the response as a JSON object for the 'json' method
                return {
                    response: {
                        json: () => Promise.resolve(openaiResponse), // Return the object directly
                        ok: true,
                        status: 200
                    },
                    selectedKeyId: 'vertex-ai',
                    modelCategory: 'Vertex'
                };

            } catch (error) {
                console.error(`Error during Vertex AI non-stream generation: ${error}`, error); // Keep error log in English
                return {
                    error: {
                        message: `Vertex AI non-stream generation failed: ${error.message}`,
                        type: 'vertex_ai_error'
                    },
                    status: 500
                };
            }
        }
    } catch (error) {
        console.error(`Error in Vertex AI proxy: ${error}`, error); // Keep error log in English
        // Clean up temporary file
        if (tempCredentialsPath) {
            try {
                const dirPath = path.dirname(tempCredentialsPath);
                await fs.rm(dirPath, { recursive: true, force: true });
                console.info(`Cleaned up temporary credentials directory: ${dirPath}`); 
            } catch (e) {
                console.warn(`Failed to delete temporary credentials directory: ${e}`); // Keep warn log in English
            }
        }
        return {
            error: {
                message: `Internal Vertex AI Proxy Error: ${error.message}`,
                type: 'vertex_internal_error'
            },
            status: 500
        };
    }
}

/**
 * Gets the list of Vertex supported models.
 * @returns {Array<string>} Array of supported model IDs.
 */
function getVertexSupportedModels() {
    // Return supported models if either authentication method is available
    return (isUsingExpressMode || VERTEX_JSON_STRING) ? VERTEX_SUPPORTED_MODELS : [];
}

/**
 * Checks if the Vertex feature is enabled (based on credentials or API key).
 * @returns {boolean} True if Vertex AI is enabled, false otherwise.
 */
function isVertexEnabled() {
    // Either service account JSON or Express API Key enables Vertex
    return !!VERTEX_JSON_STRING || !!process.env.EXPRESS_API_KEY;
}

module.exports = {
    proxyVertexChatCompletions,
    getVertexSupportedModels,
    isVertexEnabled // Export check function
};
