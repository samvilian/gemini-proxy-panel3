const express = require('express');
const requireAdminAuth = require('../middleware/adminAuth');
const configService = require('../services/configService');
const geminiKeyService = require('../services/geminiKeyService');
const fetch = require('node-fetch'); 
const { syncToGitHub } = require('../db');
const proxyPool = require('../utils/proxyPool'); // Import the proxy pool module
const router = express.Router();

// Apply admin authentication middleware to all /api/admin routes
router.use(requireAdminAuth);

// --- Helper for parsing request body (already exists in helpers.js, but useful here) ---
// Ensure express.json() middleware is applied in server.js
function parseBody(req) {
    if (!req.body) {
        throw new Error("Request body not parsed. Ensure express.json() middleware is used.");
    }
    return req.body;
}

// --- Gemini Key Management --- (/api/admin/gemini-keys)
router.route('/gemini-keys')
    .get(async (req, res, next) => {
        try {
            const keys = await geminiKeyService.getAllGeminiKeysWithUsage();
            res.json(keys);
        } catch (error) {
            next(error);
        }
    })
    .post(async (req, res, next) => {
        try {
            const { key, name } = parseBody(req);
             if (!key || typeof key !== 'string') {
                return res.status(400).json({ error: 'Request body must include a valid API key (string)' });
            }
            const result = await geminiKeyService.addGeminiKey(key, name);
            res.status(201).json({ success: true, ...result });
        } catch (error) {
             if (error.message.includes('duplicate API key')) {
                return res.status(409).json({ error: 'Cannot add duplicate API key' });
            }
            next(error);
        }
    });

router.delete('/gemini-keys/:id', async (req, res, next) => {
    try {
        const keyId = req.params.id;
        if (!keyId) {
             return res.status(400).json({ error: 'Missing key ID in path' });
        }
        await geminiKeyService.deleteGeminiKey(keyId);
        res.json({ success: true, id: keyId });
    } catch (error) {
         if (error.message.includes('not found')) {
            return res.status(404).json({ error: error.message });
        }
        next(error);
    }
});

// Base Gemini API URL
const BASE_GEMINI_URL = process.env.GEMINI_BASE_URL || 'https://generativelanguage.googleapis.com';
// Cloudflare Gateway base path
const CF_GATEWAY_BASE = 'https://gateway.ai.cloudflare.com/v1';
// Project ID regex pattern - 32 character hex string
const PROJECT_ID_REGEX = /^[0-9a-f]{32}$/i;
// Default Cloudflare Gateway project ID (Ensure this matches geminiProxyService.js or is appropriate)
const DEFAULT_PROJECT_ID = 'db16589aa22233d56fe69a2c3161fe3c';

// Helper to get the base URL for Gemini API, considering CF_GATEWAY
function getGeminiBaseUrl() {
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
                console.log(`Admin API: Using default Cloudflare Gateway: ${baseUrl}`);
            } else {
                 console.warn(`Admin API: Invalid DEFAULT_PROJECT_ID format: ${DEFAULT_PROJECT_ID}. Falling back to default Gemini URL.`);
            }
            // If invalid, fall back to default Gemini API URL (already set)
        } else {
            // Handle case 2: CF_GATEWAY contains projectId/gatewayName
            try {
                // Remove trailing slashes
                let gatewayValue = cfGateway.replace(/\/+$/, '');

                // Try to extract projectId/gatewayName pattern from anywhere in the string
                const pattern = /([0-9a-f]{32})\/([^\/\s]+)/i;
                const matches = gatewayValue.match(pattern);

                if (matches && matches.length >= 3) {
                    const projectId = matches[1];
                    const gatewayName = matches[2];

                    if (PROJECT_ID_REGEX.test(projectId)) {
                        baseUrl = `${CF_GATEWAY_BASE}/${projectId}/${gatewayName}/google-ai-studio`;
                        console.log(`Admin API: Using custom Cloudflare Gateway: ${baseUrl}`);
                    } else {
                         console.warn(`Admin API: Invalid Project ID format found in CF_GATEWAY: ${projectId}. Falling back to default Gemini URL.`);
                    }
                } else {
                    console.warn(`Admin API: CF_GATEWAY value "${cfGateway}" does not match expected format. Falling back to default Gemini URL.`);
                }
            } catch (error) {
                console.error('Admin API: Error parsing CF_GATEWAY value:', error);
                // Fall back to default URL on error (already set)
            }
        }
    }
    return baseUrl;
}

// --- Test Gemini Key --- (/api/admin/test-gemini-key)
router.post('/test-gemini-key', async (req, res, next) => {
     try {
        const { keyId, modelId } = parseBody(req);
        if (!keyId || !modelId) {
             return res.status(400).json({ error: 'Request body must include keyId and modelId' });
        }

        // Fetch the actual key from the database
        const keyInfo = await configService.getDb('SELECT api_key FROM gemini_keys WHERE id = ?', [keyId]);
        if (!keyInfo || !keyInfo.api_key) {
            return res.status(404).json({ error: `API Key with ID '${keyId}' not found or invalid.` });
        }
        const apiKey = keyInfo.api_key;

        // Fetch model category for potential usage increment
        const modelsConfig = await configService.getModelsConfig();
        const modelCategory = modelsConfig[modelId]?.category;

        const testGeminiRequestBody = { contents: [{ role: "user", parts: [{ text: "Hi" }] }] };
        const baseUrl = getGeminiBaseUrl();
        const geminiUrl = `${baseUrl}/v1beta/models/${modelId}:generateContent`;

        let testResponseStatus = 500;
        let testResponseBody = null;
        let isSuccess = false;

        try {
            // Get proxy agent
            const agent = proxyPool.getNextProxyAgent();
            const fetchOptions = {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-goog-api-key': apiKey
                },
                body: JSON.stringify(testGeminiRequestBody)
            };
            if (agent) {
                fetchOptions.agent = agent;
                console.log(`Admin API (Test Key): Sending request via proxy ${agent.proxy.href}`);
            } else {
                 console.log(`Admin API (Test Key): Sending request directly.`);
            }

            const response = await fetch(geminiUrl, fetchOptions);
            testResponseStatus = response.status;
            testResponseBody = await response.json(); // Attempt to parse JSON
            isSuccess = response.ok;

if (isSuccess) {
                 // Increment usage and sync to GitHub
                 await geminiKeyService.incrementKeyUsage(keyId, modelId, modelCategory);
            } else {
                 // Record 401/403 errors
                 if (testResponseStatus === 401 || testResponseStatus === 403) {
                     await geminiKeyService.recordKeyError(keyId, testResponseStatus);
                 }
            }

        } catch (fetchError) {
             console.error(`Error testing Gemini API key ${keyId}:`, fetchError);
             testResponseBody = { error: `Fetch error: ${fetchError.message}` };
             isSuccess = false;
             // Don't assume network error means key is bad, could be temporary
        }

        res.status(isSuccess ? 200 : testResponseStatus).json({
            success: isSuccess,
            status: testResponseStatus,
            content: testResponseBody
        });

    } catch (error) {
        // Errors from fetching keyInfo etc.
        next(error);
    }
});

// --- Get Available Gemini Models --- (/api/admin/gemini-models)
router.get('/gemini-models', async (req, res, next) => {
     try {
         // Find *any* valid key to make the models list request, without updating the rotation index
         // This prevents writing to the database and GitHub sync on page refreshes
         const availableKey = await geminiKeyService.getNextAvailableGeminiKey(null, false); // Don't update index for read-only operation
         if (!availableKey) {
             console.warn("No available Gemini key found to fetch models list.");
             return res.json([]); // Return empty list if no keys work
         }

         const baseUrl = getGeminiBaseUrl();
         const geminiUrl = `${baseUrl}/v1beta/models`;

         // Get proxy agent
         const agent = proxyPool.getNextProxyAgent();
         const fetchOptions = {
             method: 'GET',
             headers: {
                 'Content-Type': 'application/json',
                 'x-goog-api-key': availableKey.key
             }
         };
         if (agent) {
            fetchOptions.agent = agent;
            console.log(`Admin API (Get Models): Sending request via proxy ${agent.proxy.href}`);
         } else {
             console.log(`Admin API (Get Models): Sending request directly.`);
         }

         const response = await fetch(geminiUrl, fetchOptions);

         if (!response.ok) {
             const errorBody = await response.text();
             console.error(`Error fetching Gemini models list (key ${availableKey.id}): ${response.status} ${response.statusText}`, errorBody);
             // Don't mark the key as bad for a failed models list request
             return res.json([]); // Return empty on error
         }

         const data = await response.json();
         const processedModels = (data.models || [])
            .filter(model => model.name?.startsWith('models/')) // Ensure correct format
            .map((model) => ({
                 id: model.name.substring(7), // Extract ID
                 name: model.displayName || model.name.substring(7), // Prefer displayName
                 description: model.description,
                 // Add other potentially useful fields: supportedGenerationMethods, version, etc.
             }));

         res.json(processedModels);
     } catch (error) {
         console.error('Error handling /api/admin/gemini-models:', error);
         next(error);
     }
});


// --- Error Key Management ---
router.get('/error-keys', async (req, res, next) => {
    try {
        const errorKeys = await geminiKeyService.getErrorKeys();
        res.json(errorKeys);
    } catch (error) {
        next(error);
    }
});

router.post('/clear-key-error', async (req, res, next) => {
    try {
        const { keyId } = parseBody(req);
         if (!keyId || typeof keyId !== 'string') {
            return res.status(400).json({ error: 'Request body must include a valid keyId (string)' });
        }
        await geminiKeyService.clearKeyError(keyId);
        res.json({ success: true, id: keyId });
    } catch (error) {
         if (error.message.includes('not found')) {
            return res.status(404).json({ error: error.message });
        }
        next(error);
    }
});


// --- Worker Key Management --- (/api/admin/worker-keys)
router.route('/worker-keys')
    .get(async (req, res, next) => {
        try {
            const keys = await configService.getAllWorkerKeys();
            res.json(keys);
        } catch (error) {
            next(error);
        }
    })
    .post(async (req, res, next) => {
        try {
            const { key, description } = parseBody(req);
            if (!key || typeof key !== 'string' || key.trim() === '') {
                 return res.status(400).json({ error: 'Request body must include a valid non-empty string: key' });
            }
            await configService.addWorkerKey(key.trim(), description);
            res.status(201).json({ success: true, key: key.trim() });
        } catch (error) {
            if (error.message.includes('already exists')) {
                return res.status(409).json({ error: error.message });
            }
            next(error);
        }
    });

router.delete('/worker-keys/:key', async (req, res, next) => { // Use key in path param
     try {
        const keyToDelete = decodeURIComponent(req.params.key); // Decode URL component
         if (!keyToDelete) {
             return res.status(400).json({ error: 'Missing worker key in path' });
         }
        await configService.deleteWorkerKey(keyToDelete);
        res.json({ success: true, key: keyToDelete });
    } catch (error) {
         if (error.message.includes('not found')) {
            return res.status(404).json({ error: error.message });
        }
        next(error);
    }
});

router.post('/worker-keys/safety-settings', async (req, res, next) => { // Specific path for safety
    try {
        const { key, safetyEnabled } = parseBody(req);
        if (!key || typeof key !== 'string' || typeof safetyEnabled !== 'boolean') {
            return res.status(400).json({ error: 'Request body must include key (string) and safetyEnabled (boolean)' });
        }
        await configService.updateWorkerKeySafety(key, safetyEnabled);
        res.json({ success: true, key: key, safetyEnabled: safetyEnabled });
    } catch (error) {
         if (error.message.includes('not found')) {
            return res.status(404).json({ error: error.message });
        }
        next(error);
    }
});


// --- Model Configuration Management --- (/api/admin/models)
router.route('/models')
    .get(async (req, res, next) => {
        try {
            const config = await configService.getModelsConfig();
            // Convert to array format expected by UI
            const modelList = Object.entries(config).map(([id, data]) => ({ id, ...data }));
            res.json(modelList);
        } catch (error) {
            next(error);
        }
    })
    .post(async (req, res, next) => { // Add or Update
        try {
             const { id, category, dailyQuota, individualQuota } = parseBody(req);
             if (!id || !category || !['Pro', 'Flash', 'Custom'].includes(category)) {
                 return res.status(400).json({ error: 'Request body must include valid id and category (Pro, Flash, or Custom)' });
             }
             // Basic validation for quotas (more in service layer)
             const dailyQuotaNum = (dailyQuota === null || dailyQuota === undefined || dailyQuota === '') ? null : Number(dailyQuota);
             const individualQuotaNum = (individualQuota === null || individualQuota === undefined || individualQuota === '') ? null : Number(individualQuota);

             if ((dailyQuotaNum !== null && isNaN(dailyQuotaNum)) || (individualQuotaNum !== null && isNaN(individualQuotaNum))) {
                 return res.status(400).json({ error: 'Quotas must be numbers or null/empty.' });
             }

             await configService.setModelConfig(id, category, dailyQuotaNum, individualQuotaNum);
             res.status(200).json({ success: true, id, category, dailyQuota: dailyQuotaNum, individualQuota: individualQuotaNum }); // Use 200 for add/update simplicity
        } catch (error) {
             if (error.message.includes('must be a non-negative integer')) {
                return res.status(400).json({ error: error.message });
             }
            next(error);
        }
    });

router.delete('/models/:id', async (req, res, next) => { // Use ID in path
    try {
        const modelIdToDelete = decodeURIComponent(req.params.id);
         if (!modelIdToDelete) {
             return res.status(400).json({ error: 'Missing model ID in path' });
         }
        await configService.deleteModelConfig(modelIdToDelete);
        res.json({ success: true, id: modelIdToDelete });
    } catch (error) {
        if (error.message.includes('not found')) {
            return res.status(404).json({ error: error.message });
        }
        next(error);
    }
});


// --- Category Quota Management --- (/api/admin/category-quotas)
router.route('/category-quotas')
    .get(async (req, res, next) => {
        try {
            const quotas = await configService.getCategoryQuotas();
            res.json(quotas);
        } catch (error) {
            next(error);
        }
    })
    .post(async (req, res, next) => {
        try {
            const { proQuota, flashQuota } = parseBody(req);
            // Service layer handles detailed validation
             await configService.setCategoryQuotas(proQuota, flashQuota);
             res.json({ success: true, proQuota, flashQuota });
        } catch (error) {
             if (error.message.includes('must be non-negative numbers')) {
                 return res.status(400).json({ error: error.message });
             }
            next(error);
        }
    });


module.exports = router;
