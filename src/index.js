// Load environment variables from .env file FIRST
require('dotenv').config();

// Manually load VERTEX from .env before doing anything else that might need it
const vertexService = require('./services/vertexProxyService'); // Imports and triggers manual load

const express = require('express');
const path = require('path');
const cors = require('cors');
const cookieParser = require('cookie-parser');

// Import the database connection and sync function (this will also trigger initialization)
const { db } = require('./db'); 

// Import route handlers
const authRoutes = require('./routes/auth');
const adminApiRoutes = require('./routes/adminApi');
const apiV1Routes = require('./routes/apiV1');

// Import services and utils (ensure proxyPool is imported to trigger its initialization)
require('./services/geminiProxyService'); // Still need to import this for other initializations if any
const proxyPool = require('./utils/proxyPool');

// Import middleware
const requireAdminAuth = require('./middleware/adminAuth');

const app = express();
const port = process.env.PORT || 3000; // Default to 3000 if PORT not set

// --- Middleware ---

// Enable CORS for all origins (adjust for production if needed)
app.use(cors({
    origin: '*', // Allow all origins for now
    credentials: true, // Allow cookies for authenticated requests (like admin UI)
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'x-requested-with'],
    maxAge: 86400 // Cache preflight requests for 1 day
}));

// Handle OPTIONS preflight requests globally (alternative to handling in each route)
app.options('*', cors());

// Parse JSON request bodies
app.use(express.json({ limit: '100mb' }));

// Parse URL-encoded request bodies
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// Parse cookies
app.use(cookieParser());

// Serve static files from the 'public' directory
// __dirname now refers to the src directory, need to go up one level
app.use(express.static(path.join(__dirname, '..', 'public')));

// --- Basic Routes ---

// Root route: Redirects to /admin/index.html if logged in, otherwise requireAdminAuth redirects to /login.html
app.get('/', requireAdminAuth, (req, res) => {
    // If we reach here, user is authenticated by requireAdminAuth
    res.redirect('/admin/index.html');
});

// Redirect /login to the static HTML file
app.get('/login', (req, res) => {
    res.redirect('/login.html');
});

// Admin route: Protect the route and serve the static file
app.get('/admin', requireAdminAuth, (req, res) => {
    res.redirect('/admin/'); // Redirect to the directory path
});

app.use('/admin', requireAdminAuth, express.static(path.join(__dirname, '..', 'public', 'admin')));

// --- API Routes ---
app.use('/api', authRoutes); 
app.use('/api/admin', requireAdminAuth, adminApiRoutes); 
app.use('/v1', apiV1Routes); 

// --- Global Error Handler ---
app.use((err, req, res, next) => {
    console.error('Unhandled Error:', err.stack || err);
    res.status(err.status || 500).json({
        error: {
            message: err.message || 'Internal Server Error',
            type: err.type || 'unhandled_error',
            ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
        }
    });
});

// --- Start Server ---
app.listen(port, '0.0.0.0', () => {
    console.log(`Gemini Proxy Panel (Node.js version) listening on port ${port} (all interfaces)`);
    
    // Log Proxy Pool Status
    const proxyStatus = proxyPool.getProxyPoolStatus(); // Get status from proxyPool module
    if (proxyStatus.enabled) {
        console.log(`Proxy Pool: Enabled (Loaded ${proxyStatus.count} SOCKS5 proxies)`);
    } else if (proxyStatus.count > 0 && !proxyStatus.agentLoaded) {
        console.log(`Proxy Pool: Configured (${proxyStatus.count} proxies) but DISABLED (missing 'socks-proxy-agent' dependency)`);
    } else {
        console.log(`Proxy Pool: Disabled (PROXY environment variable not set or contains no valid SOCKS5 proxies)`);
    }
    
    // Log Vertex AI Status using the check function
    if (vertexService.isVertexEnabled()) {
        // Check if we're using Express Mode
        if (process.env.EXPRESS_API_KEY) {
            console.log(`Vertex AI: Enabled with Express Mode (API Key authentication, additional [v] prefixed models available)`);
        } else {
            console.log(`Vertex AI: Enabled (Service Account credentials, additional [v] prefixed models available)`);
        }
    } else {
        console.log(`Vertex AI: Disabled (VERTEX variable and EXPRESS_API_KEY not found or invalid in .env file)`);
    }
    
    // Check if running in Hugging Face Space
    if (process.env.HUGGING_FACE === '1' && process.env.SPACE_HOST) {
        const adminUrl = `https://${process.env.SPACE_HOST}/admin`;
        console.log(`Hugging Face Space Admin UI: ${adminUrl}`);
    } else {
        // Fallback for local or other environments
        console.log(`Admin UI should be available at http://localhost:${port}/admin (or the server's public address)`);
    }
});
