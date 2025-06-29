// --- Transformation logic migrated from Cloudflare Worker ---

/**
 * Parses a data URI string.
 * @param {string} dataUri - The data URI (e.g., "data:image/jpeg;base64,...").
 * @returns {{ mimeType: string; data: string } | null} Parsed data or null if invalid.
 */
function parseDataUri(dataUri) {
    if (!dataUri) return null;
	const match = dataUri.match(/^data:(.+?);base64,(.+)$/);
	if (!match) return null;
	return { mimeType: match[1], data: match[2] };
}

/**
 * 递归移除对象中的 additionalProperties 字段
 * @param {object|array} obj 
 * @returns {object|array}
 */
function stripAdditionalProperties(obj) {
    if (Array.isArray(obj)) {
        return obj.map(stripAdditionalProperties);
    } else if (obj && typeof obj === 'object') {
        const newObj = {};
        for (const key in obj) {
            if (key !== 'additionalProperties') {
                newObj[key] = stripAdditionalProperties(obj[key]);
            }
        }
        return newObj;
    }
    return obj;
}

/**
 * Transforms an OpenAI-compatible request body to the Gemini API format.
 * @param {object} requestBody - The OpenAI request body.
 * @param {string} [requestedModelId] - The specific model ID requested.
 * @param {boolean} [isSafetyEnabled=true] - Whether safety filtering is enabled for this request.
 * @returns {{ contents: any[]; systemInstruction?: any; tools?: any[] }} Gemini formatted request parts.
 */
function transformOpenAiToGemini(requestBody, requestedModelId, isSafetyEnabled = true) {
	const messages = requestBody.messages || [];
	const openAiTools = requestBody.tools;

	// 新增：记录已处理的 tool_call_id，防止死循环
	const processedToolCallIds = new Set();

	// 1. Transform Messages
	const contents = [];
	let systemInstruction = undefined;
	let systemMessageLogPrinted = false; // Add flag to track if log has been printed

	messages.forEach((msg) => {
		let role = undefined;
		let parts = [];

		// 1. Map Role
		switch (msg.role) {
			case 'user':
				role = 'user';
				break;
			case 'assistant':
				role = 'model';
				break;
			case 'system':
                // If safety is disabled OR it's a gemma model, treat system as user
                if (isSafetyEnabled === false || (requestedModelId && requestedModelId.startsWith('gemma'))) {
                    // Only print the log message for the first system message encountered
                    if (!systemMessageLogPrinted) {
                        console.log(`Safety disabled (${isSafetyEnabled}) or Gemma model detected (${requestedModelId}). Treating system message as user message.`);
                        systemMessageLogPrinted = true;
                    }
                    role = 'user';
                    // Content processing for 'user' role will happen below
                }
                // Otherwise (safety enabled and not gemma), create systemInstruction
                else {
                    if (typeof msg.content === 'string') {
                        systemInstruction = { role: "system", parts: [{ text: msg.content }] };
                    } else if (Array.isArray(msg.content)) { // Handle complex system prompts if needed
                        const textContent = msg.content.find((p) => p.type === 'text')?.text;
                        if (textContent) {
                            systemInstruction = { role: "system", parts: [{ text: textContent }] };
                        }
                    }
                    return; // Skip adding this message to 'contents' when creating systemInstruction
                }
                break; // Break for 'system' role (safety disabled/gemma case falls through to content processing)
			case 'tool':
				role = 'user'; // In Gemini, tool responses are part of the user's turn.
				try {
					const toolCallId = msg.tool_call_id;
					if (!toolCallId) {
						console.error("Error: 'tool' message is missing 'tool_call_id'. Skipping message.", msg);
						return;
					}

					// 新增：去重逻辑
					if (processedToolCallIds.has(toolCallId)) {
						console.warn(`Duplicate tool_call_id detected: ${toolCallId}, skipping repeated tool message.`);
						return;
					}
					processedToolCallIds.add(toolCallId);

					// Extract the function name from the tool_call_id, assuming the format `call_FUNCNAME_...`
					// This matches the format generated in `transformGeminiStreamChunk` and `transformGeminiResponseToOpenAI`.
					const match = toolCallId.match(/^call_([a-zA-Z0-9_-]+)_/);
					let toolName = match ? match[1] : msg.name; // Fallback to msg.name if it exists

					if (!toolName) {
						toolName = 'unknown_tool';
						console.warn(`Warning: Could not extract function name from tool_call_id: ${toolCallId} and msg.name is missing. Using 'unknown_tool' as fallback.`);
					}

					let toolOutput = msg.content;

					// Attempt to parse content as JSON, if it's a string
					if (typeof toolOutput === 'string') {
						try {
							toolOutput = JSON.parse(toolOutput);
						} catch (e) {
							// It's common for tool content to be a simple string, not JSON.
							// Gemini expects a JSON object for the 'response' field.
							// console.warn(`Tool content for ${toolName} is not JSON: "${msg.content}". Wrapping it.`);
							toolOutput = { content: toolOutput }; // Wrap the string content
						}
					} else if (toolOutput === undefined || toolOutput === null) {
						toolOutput = {}; // Ensure it's an object
					}

					parts.push({
						functionResponse: {
							name: toolName,
							response: toolOutput,
						},
					});
				} catch (e) {
					console.error(`Error processing tool message: ${e.message}. Skipping message.`);
					return; // Skip message on error
				}
				break;
			default:
				console.warn(`Unknown role encountered: ${msg.role}. Skipping message.`);
				return; // Skip unknown roles
		}

		// 2. Map Content to Parts
		// 2. Map Content to Parts
		// Handle text and image content parts
		if (typeof msg.content === 'string') {
			parts.push({ text: msg.content });
		} else if (Array.isArray(msg.content)) {
			msg.content.forEach((part) => {
				if (part.type === 'text') {
					parts.push({ text: part.text });
				} else if (part.type === 'image_url') {
					const imageUrl = part.image_url?.url;
					if (!imageUrl) {
						console.warn(`Missing url in image_url part. Skipping image part.`);
						return;
					}
					const imageData = parseDataUri(imageUrl);
					if (imageData) {
						parts.push({ inlineData: { mimeType: imageData.mimeType, data: imageData.data } });
					} else {
						console.warn(`Image URL is not a data URI: ${imageUrl}. Gemini API requires inlineData. Skipping image part.`);
					}
				}
			});
		}

		// Handle tool calls from assistant, which can coexist with content
		if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
			msg.tool_calls.forEach(toolCall => {
				try {
					parts.push({
						functionCall: {
							name: toolCall.function.name,
							args: JSON.parse(toolCall.function.arguments),
						},
					});
				} catch (e) {
					console.error(`Error parsing tool_call arguments for ${toolCall.function.name}: ${e.message}. Skipping tool call.`);
				}
			});
		}

		// Final check for unsupported content types, allowing for null content when tool_calls are present
		if (parts.length === 0 && msg.content !== null) {
			console.warn(`Unsupported content type for role ${msg.role}: ${typeof msg.content}. Skipping message.`);
			return;
		}

		// Add the transformed message to contents if it has a role and parts
		if (role && parts.length > 0) {
			contents.push({ role, parts });
		}
	});

	// 2. Transform Tools
	let geminiTools = undefined;
	if (openAiTools && Array.isArray(openAiTools) && openAiTools.length > 0) {
		const functionDeclarations = openAiTools
			.filter(tool => tool.type === 'function' && tool.function)
			.map(tool => {
                // Deep clone parameters 并递归移除 additionalProperties
                let parameters = tool.function.parameters ? JSON.parse(JSON.stringify(tool.function.parameters)) : undefined;
                if (parameters) {
                    parameters = stripAdditionalProperties(parameters);
                }
                // Remove the $schema field if it exists in the clone
                if (parameters && parameters.$schema !== undefined) {
                    delete parameters.$schema;
                    console.log(`Removed '$schema' from parameters for tool: ${tool.function.name}`);
                }
				return {
					name: tool.function.name,
					description: tool.function.description,
					parameters: parameters
				};
			});

		if (functionDeclarations.length > 0) {
			geminiTools = [{ functionDeclarations }];
		}
	}

	return { contents, systemInstruction, tools: geminiTools };
}


/**
 * Transforms a single Gemini API stream chunk into an OpenAI-compatible SSE chunk.
 * @param {object} geminiChunk - The parsed JSON object from a Gemini stream line.
 * @param {string} modelId - The model ID used for the request.
 * @returns {string | null} An OpenAI SSE data line string ("data: {...}\n\n") or null if chunk is empty/invalid.
 */
function transformGeminiStreamChunk(geminiChunk, modelId) {
    try {
        if (!geminiChunk || !geminiChunk.candidates || !geminiChunk.candidates.length) {
            if (geminiChunk?.usageMetadata) {
                return null;
            }
            console.warn("Received empty or invalid Gemini stream chunk:", JSON.stringify(geminiChunk));
            return null;
        }

        const candidate = geminiChunk.candidates[0];
        let sseEvents = []; // Array to hold all SSE event strings to be returned

        // --- Part 1: Handle 'thought' parts as custom SSE events ---
        if (candidate.content?.parts?.length > 0) {
            const thoughtParts = candidate.content.parts.filter((part) => part.thought !== undefined);
            if (thoughtParts.length > 0) {
                thoughtParts.forEach(part => {
                    const thought = part.thought;
                    let thoughtEventData;
                    if (thought.toolCode) {
                        thoughtEventData = { type: 'tool_code', content: thought.toolCode };
                    } else if (thought.placeholder !== undefined) { // Check for placeholder existence
                        thoughtEventData = { type: 'placeholder' };
                    }
                    
                    if (thoughtEventData) {
                        const eventString = `event: thought_process\ndata: ${JSON.stringify(thoughtEventData)}\n\n`;
                        sseEvents.push(eventString);
                    }
                });
            }
        }

        // --- Part 2: Handle 'text' and 'functionCall' parts as standard OpenAI chunks ---
        let contentText = null;
        let toolCalls = undefined;

        if (candidate.content?.parts?.length > 0) {
            const textParts = candidate.content.parts.filter((part) => part.text !== undefined);
            const functionCallParts = candidate.content.parts.filter((part) => part.functionCall !== undefined);

            if (textParts.length > 0) {
                contentText = textParts.map((part) => part.text).join("");
            }

            if (functionCallParts.length > 0) {
                toolCalls = functionCallParts.map((part, index) => ({
                    index: index,
                    id: `call_${part.functionCall.name}_${Date.now()}_${index}`,
                    type: "function",
                    function: {
                        name: part.functionCall.name,
                        arguments: JSON.stringify(part.functionCall.args || {}),
                    },
                }));
            }
        }

        let finishReason = candidate.finishReason;
        if (finishReason === "STOP") finishReason = "stop";
        else if (finishReason === "MAX_TOKENS") finishReason = "length";
        else if (finishReason === "SAFETY" || finishReason === "RECITATION") finishReason = "content_filter";
        else if (finishReason === "TOOL_CALLS" || (toolCalls && toolCalls.length > 0 && finishReason !== 'stop' && finishReason !== 'length')) {
            finishReason = "tool_calls";
        } else if (finishReason && finishReason !== "FINISH_REASON_UNSPECIFIED" && finishReason !== "OTHER") {
            // Keep known reasons
        } else {
            finishReason = null;
        }

        const delta = {};
        if (candidate.content?.role && (contentText !== null || (toolCalls && toolCalls.length > 0))) {
            delta.role = candidate.content.role === 'model' ? 'assistant' : candidate.content.role;
        }

        if (toolCalls && toolCalls.length > 0) {
            delta.tool_calls = toolCalls;
            if (contentText === null) {
                delta.content = null;
            } else {
                 delta.content = contentText;
            }
        } else if (contentText !== null) {
            delta.content = contentText;
        }

        if (Object.keys(delta).length > 0 || finishReason) {
            const openaiChunk = {
                id: `chatcmpl-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`,
                object: "chat.completion.chunk",
                created: Math.floor(Date.now() / 1000),
                model: modelId,
                choices: [
                    {
                        index: candidate.index || 0,
                        delta: delta,
                        finish_reason: finishReason,
                        logprobs: null,
                    },
                ],
            };
            sseEvents.push(`data: ${JSON.stringify(openaiChunk)}\n\n`);
        }

        return sseEvents.length > 0 ? sseEvents.join('') : null;

    } catch (e) {
        console.error("Error transforming Gemini stream chunk:", e, "Chunk:", JSON.stringify(geminiChunk));
        const errorChunk = {
            id: `chatcmpl-error-${Date.now()}`,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: modelId,
            choices: [{ index: 0, delta: { content: `[Error transforming chunk: ${e.message}]` }, finish_reason: 'error' }]
        };
        return `data: ${JSON.stringify(errorChunk)}\n\n`;
    }
}


/**
 * Transforms a complete (non-streaming) Gemini API response into an OpenAI-compatible format.
 * @param {object} geminiResponse - The parsed JSON object from the Gemini API response.
 * @param {string} modelId - The model ID used for the request.
 * @returns {string} A JSON string representing the OpenAI-compatible response.
 */
function transformGeminiResponseToOpenAI(geminiResponse, modelId) {
    try {
        if (!geminiResponse.candidates || geminiResponse.candidates.length === 0) {
            let errorMessage = "Gemini response missing candidates.";
            let finishReason = "error";

            if (geminiResponse.promptFeedback?.blockReason) {
                errorMessage = `Request blocked by Gemini: ${geminiResponse.promptFeedback.blockReason}.`;
                finishReason = "content_filter";
                 console.warn(`Gemini request blocked: ${geminiResponse.promptFeedback.blockReason}`, JSON.stringify(geminiResponse.promptFeedback));
            } else {
                console.error("Invalid Gemini response structure:", JSON.stringify(geminiResponse));
            }

            const errorResponse = {
                id: `chatcmpl-error-${Date.now()}`,
                object: "chat.completion",
                created: Math.floor(Date.now() / 1000),
                model: modelId,
                choices: [{
                    index: 0,
                    message: { role: "assistant", content: errorMessage },
                    finish_reason: finishReason,
                    logprobs: null,
                }],
                usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            };
            return JSON.stringify(errorResponse);
        }

        const candidate = geminiResponse.candidates[0];
        let contentText = null;
        let toolCalls = undefined;
        let thoughtProcess = [];

        if (candidate.content?.parts?.length > 0) {
            const thoughtParts = candidate.content.parts.filter((part) => part.thought !== undefined);
            if (thoughtParts.length > 0) {
                thoughtProcess = thoughtParts.map(part => {
                    const thought = part.thought;
                    if (thought.toolCode) {
                        return { type: 'tool_code', content: thought.toolCode };
                    } else if (thought.placeholder !== undefined) {
                        return { type: 'placeholder' };
                    }
                    return null;
                }).filter(t => t !== null);
            }

            const textParts = candidate.content.parts.filter((part) => part.text !== undefined);
            const functionCallParts = candidate.content.parts.filter((part) => part.functionCall !== undefined);

            if (textParts.length > 0) {
                contentText = textParts.map((part) => part.text).join("");
            }

            if (functionCallParts.length > 0) {
                toolCalls = functionCallParts.map((part, index) => ({
                    id: `call_${part.functionCall.name}_${Date.now()}_${index}`,
                    type: "function",
                    function: {
                        name: part.functionCall.name,
                        arguments: JSON.stringify(part.functionCall.args || {}),
                    },
                }));
            }
        }

        let finishReason = candidate.finishReason;
        if (finishReason === "STOP") finishReason = "stop";
        else if (finishReason === "MAX_TOKENS") finishReason = "length";
        else if (finishReason === "SAFETY" || finishReason === "RECITATION") finishReason = "content_filter";
        else if (finishReason === "TOOL_CALLS") finishReason = "tool_calls";
        else if (toolCalls && toolCalls.length > 0) {
            finishReason = "tool_calls";
        } else if (finishReason && finishReason !== "FINISH_REASON_UNSPECIFIED" && finishReason !== "OTHER") {
            // Keep known reasons
        } else {
             finishReason = null;
        }

        if (contentText === null && !toolCalls && candidate.finishReason === "SAFETY") {
             console.warn("Gemini response finished due to SAFETY, content might be missing.");
             contentText = "[Content blocked due to safety settings]";
             finishReason = "content_filter";
        } else if (candidate.finishReason === "RECITATION") {
             console.warn("Gemini response finished due to RECITATION.");
             finishReason = "content_filter";
        }

        const message = { role: "assistant" };
        if (toolCalls && toolCalls.length > 0) {
             message.tool_calls = toolCalls;
             message.content = contentText !== null ? contentText : null;
        } else {
             message.content = contentText;
        }
         if (message.content === undefined && !message.tool_calls) {
            message.content = null;
         }

        const usage = {
            prompt_tokens: geminiResponse.usageMetadata?.promptTokenCount || 0,
            completion_tokens: geminiResponse.usageMetadata?.candidatesTokenCount || 0,
            total_tokens: geminiResponse.usageMetadata?.totalTokenCount || 0,
        };

        const openaiResponse = {
            id: `chatcmpl-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: modelId,
            choices: [
                {
                    index: candidate.index || 0,
                    message: message,
                    finish_reason: finishReason,
                    logprobs: null,
                },
            ],
            usage: usage,
            system_fingerprint: null
        };

        if (thoughtProcess.length > 0) {
            openaiResponse.x_gemini_thought_process = thoughtProcess;
        }

        return JSON.stringify(openaiResponse);

    } catch (e) {
        console.error("Error transforming Gemini non-stream response:", e, "Response:", JSON.stringify(geminiResponse));
        const errorResponse = {
            id: `chatcmpl-error-${Date.now()}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: modelId,
            choices: [{
                index: 0,
                message: { role: "assistant", content: `Error processing Gemini response: ${e.message}` },
                finish_reason: "error",
                logprobs: null,
            }],
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        };
        return JSON.stringify(errorResponse);
    }
}


module.exports = {
    parseDataUri,
    transformOpenAiToGemini,
    transformGeminiStreamChunk,
    transformGeminiResponseToOpenAI,
};
