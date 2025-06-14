const fetch = require('node-fetch');
const { Readable } = require('stream');
const { URL } = require('url'); // Import URL for parsing remains relevant for potential future URL parsing
const { syncToGitHub } = require('../db');
const configService = require('./configService');
const geminiKeyService = require('./geminiKeyService');
const transformUtils = require('../utils/transform');
const proxyPool = require('../utils/proxyPool'); // Import the new proxy pool module


// Base Gemini API URL
const BASE_GEMINI_URL = process.env.GEMINI_BASE_URL || 'https://generativelanguage.googleapis.com';
// Cloudflare Gateway base path
const CF_GATEWAY_BASE = 'https://gateway.ai.cloudflare.com/v1';
// Project ID regex pattern - 32 character hex string
const PROJECT_ID_REGEX = /^[0-9a-f]{32}$/i;
// Default Cloudflare Gateway project ID (Replace with your actual default if needed)
const DEFAULT_PROJECT_ID = 'db16589aa22233d56fe69a2c3161fe3c';

async function proxyChatCompletions(openAIRequestBody, workerApiKey, stream, thinkingBudget) {
    // Check if KEEPALIVE mode is enabled
    const keepAliveEnabled = process.env.KEEPALIVE === '1';
    
    const requestedModelId = openAIRequestBody?.model;

    if (!requestedModelId) {
        return { error: { message: "Missing 'model' field in request body" }, status: 400 };
    }
    if (!openAIRequestBody.messages || !Array.isArray(openAIRequestBody.messages)) {
        return { error: { message: "Missing or invalid 'messages' field in request body" }, status: 400 };
    }

    const MAX_RETRIES = 3;
    let lastError = null;
    let lastErrorStatus = 500;
    let modelInfo;
    let modelCategory;
    let isSafetyEnabled;
    let modelsConfig;

    try {
        // Fetch model config and safety settings once before the loop
        [modelsConfig, isSafetyEnabled] = await Promise.all([
            configService.getModelsConfig(),
            configService.getWorkerKeySafetySetting(workerApiKey) // Get safety setting for this worker key
        ]);
        
        // Check if web search functionality needs to be added
        // 1. Via web_search parameter or 2. Using a model ending with -search
        const isSearchModel = requestedModelId.endsWith('-search');
        const actualModelId = isSearchModel ? requestedModelId.replace('-search', '') : requestedModelId;
        
        // If KEEPALIVE is enabled, this is a streaming request, and safety is disabled, we'll handle it specially
        const useKeepAlive = !isSafetyEnabled && keepAliveEnabled && stream;
    
        // If using keepalive, we'll make a non-streaming request to Gemini but send streaming responses to client
        const actualStreamMode = useKeepAlive ? false : stream;

        // If it's a search model, use the original model ID to find model info
        const modelLookupId = isSearchModel ? actualModelId : requestedModelId;
        modelInfo = modelsConfig[modelLookupId];
        if (!modelInfo) {
            return { error: { message: `Model '${modelLookupId}' is not configured in the proxy.` }, status: 400 };
        }
        modelCategory = modelInfo.category;

        // --- Retry Loop ---
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            let selectedKey;
            let forceNewKey = false; // Flag to force getting a new key on retry
            try {
                // 1. Get Key inside the loop for each attempt
                // If it's a search model, use the original model ID to get the API key
                const keyModelId = isSearchModel ? actualModelId : requestedModelId;
                
                // If previous attempt had an empty response, force getting a new key by calling getNextAvailableGeminiKey
                selectedKey = await geminiKeyService.getNextAvailableGeminiKey(keyModelId);

                // 2. Validate Key
                if (!selectedKey) {
                    console.error(`Attempt ${attempt}: No available Gemini API Key found.`);
                    if (attempt === 1) {
                        // If no key on first try, return 503 immediately
                        return { error: { message: "No available Gemini API Key configured or all keys are currently rate-limited/invalid." }, status: 503 };
                    } else {
                        // If no key on subsequent tries (after 429), return the last recorded 429 error
                         console.error(`Attempt ${attempt}: No more keys to try after previous 429.`);
                         return { error: lastError, status: lastErrorStatus };
                    }
                }

                console.log(`Attempt ${attempt}: Proxying request for model: ${requestedModelId}, Category: ${modelCategory}, KeyID: ${selectedKey.id}, Safety: ${isSafetyEnabled}`);

                // 3. Transform Request Body (remains the same)
                const { contents, systemInstruction, tools: geminiTools } = transformUtils.transformOpenAiToGemini(
                    openAIRequestBody,
                    requestedModelId,
                    isSafetyEnabled // Pass safety setting to transformer
                );

                if (contents.length === 0 && !systemInstruction) {
                    return { error: { message: "Request must contain at least one user or assistant message." }, status: 400 };
                }

                const geminiRequestBody = {
                    contents: contents,
                    generationConfig: {
                        ...(openAIRequestBody.temperature !== undefined && { temperature: openAIRequestBody.temperature }),
                        ...(openAIRequestBody.top_p !== undefined && { topP: openAIRequestBody.top_p }),
                        ...(openAIRequestBody.max_tokens !== undefined && { maxOutputTokens: openAIRequestBody.max_tokens }),
                        ...(openAIRequestBody.stop && { stopSequences: Array.isArray(openAIRequestBody.stop) ? openAIRequestBody.stop : [openAIRequestBody.stop] }),
                        ...(thinkingBudget !== undefined && { thinkingConfig: { thinkingBudget: thinkingBudget } }),
                    },
                    ...(geminiTools && { tools: geminiTools }),
                    ...(systemInstruction && { systemInstruction: systemInstruction }),
                };

                if (openAIRequestBody.web_search === 1 || isSearchModel) {
                    console.log(`Web search enabled for this request (${isSearchModel ? 'model-based' : 'parameter-based'})`);
                    
                    // Create Google Search tool
                    const googleSearchTool = {
                        googleSearch: {}
                    };
                    
                    // Add to existing tools or create a new tools array
                    if (geminiRequestBody.tools) {
                        geminiRequestBody.tools = [...geminiRequestBody.tools, googleSearchTool];
                    } else {
                        geminiRequestBody.tools = [googleSearchTool];
                    }
                    
                    // Add a prompt at the end of the request to encourage the model to use search tools
                    geminiRequestBody.contents.push({
                        role: 'user',
                        parts: [{ text: '(Use search tools to get the relevant information and complete this request.)' }]
                    });
                }

                if (!isSafetyEnabled) {
                    geminiRequestBody.safetySettings = [
                        { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'OFF' }, 
                        { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'OFF' }, 
                        { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'OFF' }, 
                        { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'OFF' }, 
                        { category: 'HARM_CATEGORY_CIVIC_INTEGRITY', threshold: 'BLOCK_NONE' }, 
                    ];
                     console.log("Applying safety settings.");
                }

                // 4. Prepare and Send Request to Gemini
                // If keepalive is enabled and original request was streaming, use non-streaming API
                const apiAction = actualStreamMode ? 'streamGenerateContent' : 'generateContent';
                
                // Determine Base URL based on CF_GATEWAY environment variable
                let baseUrl = BASE_GEMINI_URL; // Default to standard Gemini URL
                const cfGateway = process.env.CF_GATEWAY;

                // Return default URL if CF_GATEWAY is not set
                if (!cfGateway) {
                    // Use default Gemini API URL (already set)
                } else {
                    // Handle case 1: CF_GATEWAY = "1" (use default project ID)
                    if (cfGateway === '1') {
                        // Validate default project ID format
                        if (PROJECT_ID_REGEX.test(DEFAULT_PROJECT_ID)) {
                            // Only use default Cloudflare Gateway if project ID format is valid
                            baseUrl = `${CF_GATEWAY_BASE}/${DEFAULT_PROJECT_ID}/gemini/google-ai-studio`;
                            console.log(`Using default Cloudflare Gateway: ${baseUrl}`);
                        } else {
                             console.warn(`Invalid DEFAULT_PROJECT_ID format: ${DEFAULT_PROJECT_ID}. Falling back to default Gemini URL.`);
                        }
                        // If invalid, fall back to default Gemini API URL (already set)
                    } else {
                        // Handle case 2: CF_GATEWAY contains projectId/gatewayName
                        try {
                            // Remove trailing slashes
                            let gatewayValue = cfGateway.replace(/\/+$/, '');

                            // Try to extract projectId/gatewayName pattern from anywhere in the string
                            // This will work for both full URLs and direct format strings like "projectId/gatewayName"
                            const pattern = /([0-9a-f]{32})\/([^\/\s]+)/i;
                            const matches = gatewayValue.match(pattern);

                            if (matches && matches.length >= 3) {
                                const projectId = matches[1];
                                const gatewayName = matches[2];

                                if (PROJECT_ID_REGEX.test(projectId)) {
                                    baseUrl = `${CF_GATEWAY_BASE}/${projectId}/${gatewayName}/google-ai-studio`;
                                    console.log(`Using custom Cloudflare Gateway: ${baseUrl}`);
                                } else {
                                     console.warn(`Invalid Project ID format found in CF_GATEWAY: ${projectId}. Falling back to default Gemini URL.`);
                                }
                            } else {
                                console.warn(`CF_GATEWAY value "${cfGateway}" does not match expected format (e.g., 'projectId/gatewayName' or full URL). Falling back to default Gemini URL.`);
                            }
                        } catch (error) {
                            console.error('Error parsing CF_GATEWAY value:', error);
                            // Fall back to default URL on error (already set)
                        }
                    }
                    // For any other value or format issue of CF_GATEWAY, keep using default Gemini API URL
                }

                // Build complete API URL using the determined base URL
                // Use actualModelId instead of requestedModelId with -search suffix
                const geminiUrl = `${baseUrl}/v1beta/models/${actualModelId}:${apiAction}`;

                const geminiRequestHeaders = {
                    'Content-Type': 'application/json',
                    'User-Agent': `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36`,
                    'X-Accel-Buffering': 'no',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'x-goog-api-key': selectedKey.key
                };

                // Get the next proxy agent for this request
                const agent = proxyPool.getNextProxyAgent(); // Use function from imported module

                // Log proxy usage here if an agent is obtained
                const logSuffix = agent ? ` via proxy ${agent.proxy.href}` : ''; // Get proxy URL from agent if available
                console.log(`Attempt ${attempt}: Sending ${actualStreamMode ? 'streaming' : 'non-streaming'} request to Gemini URL: ${geminiUrl}${logSuffix}`);
                
                // Log if using keepalive mode
                if (keepAliveEnabled && stream) {
                    if (useKeepAlive) {
                        console.log(`Using KEEPALIVE mode: Client expects stream but sending non-streaming request to Gemini (Safety disabled)`);
                    } else {
                        console.log(`KEEPALIVE is enabled but safety is also enabled. Using normal streaming mode.`);
                    }
                }

                const fetchOptions = { // Create options object
                    method: 'POST',
                    headers: geminiRequestHeaders,
                    body: JSON.stringify(geminiRequestBody),
                    size: 100 * 1024 * 1024,
                    timeout: 300000
                };

                // Add agent to options only if it's defined
                if (agent) {
                    fetchOptions.agent = agent;
                }

                const geminiResponse = await fetch(geminiUrl, fetchOptions); // Use fetchOptions

                // 5. Handle Gemini Response Status and Errors
                if (!geminiResponse.ok) {
                    const errorBodyText = await geminiResponse.text();
                    console.error(`Attempt ${attempt}: Gemini API error: ${geminiResponse.status} ${geminiResponse.statusText}`, errorBodyText);

                    lastErrorStatus = geminiResponse.status; // Store status
                    try {
                        lastError = JSON.parse(errorBodyText).error || { message: errorBodyText }; // Try parsing, fallback to text
                    } catch {
                        lastError = { message: errorBodyText };
                    }
                     // Add type and code if not present from Gemini
                    if (!lastError.type) lastError.type = `gemini_api_error_${geminiResponse.status}`;
                    if (!lastError.code) lastError.code = geminiResponse.status;


                    // Handle specific errors impacting key status
                    if (geminiResponse.status === 429) {
                        // Pass the full parsed error object (lastError) which may contain quotaId
                        console.log(`429 error details: ${JSON.stringify(lastError)}`);
                        
                        // Record 429 for the key - use actualModelId for consistent counting
                        geminiKeyService.handle429Error(selectedKey.id, modelCategory, actualModelId, lastError)
                            .catch(err => console.error(`Error handling 429 for key ${selectedKey.id} in background:`, err));

                        // If not the last attempt, continue to the next key
                        if (attempt < MAX_RETRIES) {
                            console.warn(`Attempt ${attempt}: Received 429, trying next key...`);
                            continue; // Go to the next iteration of the loop
                        } else {
                            console.error(`Attempt ${attempt}: Received 429, but max retries reached.`);
                            // Fall through to return the last recorded 429 error after the loop
                        }
                    } else if (geminiResponse.status === 401 || geminiResponse.status === 403) {
                        // Record persistent error for the key
                        geminiKeyService.recordKeyError(selectedKey.id, geminiResponse.status)
                             .catch(err => console.error(`Error recording key error ${geminiResponse.status} for key ${selectedKey.id} in background:`, err));
                        // Do not retry for 401/403, break and return this error
                        break;
                    } else {
                         // For other errors (400, 500, etc.), don't retry, break and return the error
                         console.error(`Attempt ${attempt}: Received non-retryable error ${geminiResponse.status}.`);
                         break;
                    }
                } else {
                    // 6. Process Successful Response
                    console.log(`Attempt ${attempt}: Request successful with key ${selectedKey.id}.`);
                    // Increment usage count for the actual model ID, not the -search version
                    geminiKeyService.incrementKeyUsage(selectedKey.id, actualModelId, modelCategory)
                          .catch(err => console.error(`Error incrementing usage for key ${selectedKey.id} in background:`, err));

                    // For KEEPALIVE mode with streaming client request
                    if (useKeepAlive) {
                        // Get the complete non-streaming response
                        const geminiResponseData = await geminiResponse.json();
                        
                        // Check if it's an empty response (finishReason is OTHER and no content)
                        const isEmptyResponse = geminiResponseData.candidates && 
                                               geminiResponseData.candidates[0] && 
                                               geminiResponseData.candidates[0].finishReason === "OTHER" && 
                                               (!geminiResponseData.candidates[0].content || 
                                                !geminiResponseData.candidates[0].content.parts || 
                                                geminiResponseData.candidates[0].content.parts.length === 0);
                        
                        if (isEmptyResponse && attempt < MAX_RETRIES) {
                            console.log(`Detected empty response (finishReason: OTHER), attempting retry #${attempt + 1} with a new key...`);
                            // Skip this key on next attempt
                            forceNewKey = true;
                            continue; // Continue to the next attempt
                        }
                        
                        console.log(`Chat completions call completed successfully.`);

                        // Return the complete response data, let apiV1.js handle keepalive and response sending
                        return {
                            response: geminiResponseData, // Directly return the parsed JSON data
                            selectedKeyId: selectedKey.id,
                            modelCategory: modelCategory,
                            isKeepAlive: true, // Mark this as a keepalive mode response
                            requestedModelId: requestedModelId // Pass modelId for subsequent use
                        };
                    } else {
                        // For non-KEEPALIVE mode (正常流式)，不要提前消费 response.body，直接返回
                        console.log(`Chat completions call completed successfully.`);
                        return {
                            response: geminiResponse,
                            selectedKeyId: selectedKey.id,
                            modelCategory: modelCategory
                        };
                    }
                }

            } catch (fetchError) {
                 // Catch network errors or other errors during fetch/key selection within an attempt
                 console.error(`Attempt ${attempt}: Error during proxy call:`, fetchError);
                 lastError = { message: `Internal Proxy Error during attempt ${attempt}: ${fetchError.message}`, type: 'proxy_internal_error' };
                 lastErrorStatus = 500;
                 // If a network error occurs, break the loop, don't retry immediately
                 break;
            }
        } // --- End Retry Loop ---

        // If the loop finished without returning a success or a specific non-retryable error,
        // it means all retries resulted in 429 or we broke due to an error. Return the last recorded error.
        console.error(`All ${MAX_RETRIES} attempts failed. Returning last recorded error (Status: ${lastErrorStatus}).`);
        return { error: lastError, status: lastErrorStatus };


    } catch (initialError) {
         // Catch errors happening *before* the loop starts (e.g., getting initial config)
        console.error("Error before starting proxy attempts:", initialError);
        return {
            error: {
                message: `Internal Proxy Error: ${initialError.message}`,
                type: 'proxy_internal_error'
            },
            status: 500
        };
    }
}

module.exports = {
    proxyChatCompletions,
    // getProxyPoolStatus is no longer needed here, it's in proxyPool.js
};
