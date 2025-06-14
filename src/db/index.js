const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');
const GitHubSync = require('../utils/githubSync');

// Construct the database path
let dataDir;

// Use /home/user/data directory on Hugging Face Space
if (process.env.HUGGING_FACE === '1') {
  dataDir = '/home/user/data';
  console.log(`Using Hugging Face persistent data directory: ${dataDir}`);
} else {
  dataDir = path.resolve(__dirname, '..', '..', 'data');
}

if (!fs.existsSync(dataDir)) {
  console.log(`Creating data directory: ${dataDir}`);
  try {
    fs.mkdirSync(dataDir, { recursive: true });
  } catch (err) {
    console.error(`Error creating data directory: ${err.message}`);
    console.error('Will attempt to use ./data as fallback');
    dataDir = path.resolve(__dirname, '..', '..', 'data');
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir, { recursive: true });
    }
  }
}

const dbPath = path.resolve(dataDir, 'database.db');
console.log(`Database path: ${dbPath}`); // Log the path for debugging

// Initialize GitHub sync if configured
const githubProject = process.env.GITHUB_PROJECT;
const githubToken = process.env.GITHUB_PROJECT_PAT;
const githubEncryptKey = process.env.GITHUB_ENCRYPT_KEY;
let githubSync = null;

if (githubProject && githubToken) {
  console.log(`GitHub sync configured for repository: ${githubProject}`);
  githubSync = new GitHubSync(githubProject, githubToken, dbPath, githubEncryptKey);
  
  if (githubEncryptKey && githubEncryptKey.length >= 32) {
    console.log('GitHub data encryption enabled, using AES-256-CBC algorithm');
  } else if (githubEncryptKey) {
    console.warn('GitHub encryption key length is insufficient, requires at least 32 characters, data will be stored unencrypted');
  } else {
    console.log('GitHub data encryption not enabled, data will be stored unencrypted');
  }
}

// Initialize the database connection
// The OPEN_READWRITE | OPEN_CREATE flag ensures the file is created if it doesn't exist.
const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE, async (err) => {
  if (err) {
    console.error('Error opening database:', err.message);
    throw err; // Throw error to stop the application if DB connection fails
  } else {
    console.log('Connected to the SQLite database.');
    
    // Initialize database schema first
    await new Promise((resolve, reject) => {
      initializeDatabase((err) => {
        if (err) reject(err);
        else resolve();
      });
    });
    
    // Try to download database from GitHub if configured
    if (githubSync) {
      try {
        await githubSync.downloadDatabase();
      } catch (err) {
        console.error('Failed to download database from GitHub:', err.message);
        console.log('Continuing with local database...');
      }
    }
  }
});

// Function to trigger GitHub sync (now always delayed)
async function syncToGitHub() {
  if (!githubSync) {
    return false;
  }

  try {
    // Schedule the delayed sync
    await githubSync.scheduleSync();
    return true; // Indicate scheduling was successful
  } catch (err) {
    console.error('Failed to schedule GitHub sync:', err.message);
    return false;
  }
}

// SQL statements to create tables (if they don't exist)
const createTablesSQL = `
  CREATE TABLE IF NOT EXISTS gemini_keys (
    id TEXT PRIMARY KEY,
    api_key TEXT NOT NULL UNIQUE,
    name TEXT,
    usage_date TEXT,
    model_usage TEXT DEFAULT '{}',       -- Store as JSON string
    category_usage TEXT DEFAULT '{}',    -- Store as JSON string
    error_status INTEGER,               -- 401, 403, or NULL
    consecutive_429_counts TEXT DEFAULT '{}', -- Store as JSON string
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS worker_keys (
    api_key TEXT PRIMARY KEY,
    description TEXT,
    safety_enabled INTEGER DEFAULT 1,  -- 1 for true, 0 for false
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS models_config (
    model_id TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK(category IN ('Pro', 'Flash', 'Custom')),
    daily_quota INTEGER,                -- NULL means unlimited
    individual_quota INTEGER            -- NULL means no individual limit
  );

  CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT                           -- Can store JSON strings or simple values
  );

  -- Initialize default category quotas if not present
  INSERT OR IGNORE INTO settings (key, value) VALUES
    ('category_quotas', '{"proQuota": 50, "flashQuota": 1500}');

  -- Initialize gemini_key_list if not present (as an empty JSON array)
  INSERT OR IGNORE INTO settings (key, value) VALUES
    ('gemini_key_list', '[]');

  -- Initialize gemini_key_index if not present
  INSERT OR IGNORE INTO settings (key, value) VALUES
    ('gemini_key_index', '0');

  -- Add other default settings as needed, e.g., last used key ID
  INSERT OR IGNORE INTO settings (key, value) VALUES
    ('last_used_gemini_key_id', '');
`;

// Function to initialize the database schema
function initializeDatabase(callback) {
  db.exec(createTablesSQL, (err) => {
    if (err) {
      console.error('Error creating database tables:', err.message);
      if (callback) callback(err);
    } else {
      console.log('Database tables checked/created successfully.');
      // You might seed initial data here if necessary
      if (callback) callback(null);
    }
  });
}

// Function to safely close the database connection
function closeDatabase() {
  db.close((err) => {
    if (err) {
      console.error('Error closing database:', err.message);
    } else {
      console.log('Database connection closed.');
    }
  });
}

// Gracefully close the database on application exit
process.on('SIGINT', () => {
  closeDatabase();
  process.exit(0);
});

process.on('SIGTERM', () => {
    closeDatabase();
    process.exit(0);
});

// Export the database connection instance and sync functions
module.exports = {
  db,
  syncToGitHub
};
