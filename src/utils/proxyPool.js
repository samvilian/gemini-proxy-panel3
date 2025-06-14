let SocksProxyAgent; // Declare variable
try {
    SocksProxyAgent = require('socks-proxy-agent').SocksProxyAgent; // Try importing
} catch (e) {
    console.warn("Optional dependency 'socks-proxy-agent' not found. SOCKS5 proxy functionality will be unavailable unless this dependency is installed.");
    SocksProxyAgent = null; // Set to null if import fails
}

let proxies = [];
let currentProxyIndex = 0;

function initializeProxyPool() {
    proxies = []; // Reset proxies on re-initialization
    currentProxyIndex = 0;
    const proxyEnv = process.env.PROXY;
    if (proxyEnv) {
        proxies = proxyEnv.split(',')
            .map(proxyStr => proxyStr.trim())
            .filter(proxyStr => {
                if (proxyStr.startsWith('socks5://')) {
                    return true;
                }
                if (proxyStr) { // Log invalid format only if non-empty string
                    console.warn(`Invalid proxy format skipped: "${proxyStr}". Only socks5:// is supported.`);
                }
                return false;
            });

        if (proxies.length > 0) {
             // This log will now be printed by index.js using getProxyPoolStatus
            // console.log(`Initialized proxy pool with ${proxies.length} SOCKS5 proxies.`);
        } else {
             // This log will now be printed by index.js using getProxyPoolStatus
            // console.log('PROXY environment variable found but contains no valid SOCKS5 proxies.');
        }
    } else {
         // This log will now be printed by index.js using getProxyPoolStatus
        // console.log('PROXY environment variable not set. No proxy will be used.');
    }
}

function getNextProxyAgent() {
    if (proxies.length === 0 || !SocksProxyAgent) {
        return undefined; // No proxies configured or agent not available
    }
    const proxyUrl = proxies[currentProxyIndex];
    currentProxyIndex = (currentProxyIndex + 1) % proxies.length; // Rotate index
    try {
        // Log proxy usage within the service where it's called for better context
        // console.log(`Using proxy: ${proxyUrl}`); 
        return new SocksProxyAgent(proxyUrl);
    } catch (e) {
        console.error(`Error creating proxy agent for ${proxyUrl}:`, e);
        return undefined; // Return undefined if agent creation fails
    }
}

// Function to get the status of the proxy pool
function getProxyPoolStatus() {
    const enabled = proxies.length > 0 && !!SocksProxyAgent; // Enabled if proxies exist AND agent is loaded
    return {
        enabled: enabled,
        count: proxies.length,
        agentLoaded: !!SocksProxyAgent // Explicitly indicate if the agent dependency loaded
    };
}

// Initialize the proxy pool when the module loads
initializeProxyPool();

module.exports = {
    initializeProxyPool, // Export for potential re-initialization if needed
    getNextProxyAgent,
    getProxyPoolStatus,
};
