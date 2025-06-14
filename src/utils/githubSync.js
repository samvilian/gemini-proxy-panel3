const { Octokit } = require('@octokit/rest');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class GitHubSync {
  constructor(repoName, token, dbPath, encryptKey) {
    this.repoName = repoName;
    this.token = token;
    this.dbPath = dbPath;
    this.encryptKey = encryptKey;
    
    // Parse GitHub repo owner and name
    const repoNameParts = this.repoName.split('/');
    if (repoNameParts.length !== 2 || !repoNameParts[0] || !repoNameParts[1]) {
      console.error(`Invalid GitHub repository format: "${repoName}", should be "username/repo-name" format`);
      this.isValid = false;
    } else {
      this.owner = repoNameParts[0];
      this.repo = repoNameParts[1];
      this.isValid = true;
      
      // Initialize Octokit with the token
      this.octokit = new Octokit({
        auth: this.token
      });
    }

    this.initialSyncCompleted = false;
    
    // Sync scheduling variables
    this.pendingSync = false;
    this.syncTimer = null;
    this.syncDelay = 300000; // 5 minute delay
  }

  // Check if GitHub sync is configured and enabled
  isConfigured() {
    return this.isValid && this.repoName && this.token && this.owner && this.repo;
  }

  // Check if encryption is configured
  isEncryptionEnabled() {
    return !!this.encryptKey && this.encryptKey.length >= 32;
  }

  // Check if a buffer appears to be encrypted with our format
  isEncryptedData(data) {
    // A simple check to determine if data is likely encrypted:
    // Our encrypted format has a 16-byte IV at the beginning,
    // and encrypted SQLite databases won't start with the standard SQLite header
    if (!data || data.length < 20) return false;
    
    // If encryption is enabled and the data doesn't match SQLite format
    // it's likely encrypted
    if (this.isEncryptionEnabled()) {
      // SQLite databases start with "SQLite format 3\0"
      const sqliteHeader = Buffer.from("SQLite format 3\0");
      const header = data.slice(0, 16);
      
      // If the first 16 bytes are not the SQLite header, it might be an IV
      // which suggests the data is encrypted
      if (Buffer.compare(header.slice(0, sqliteHeader.length), sqliteHeader) !== 0) {
        return true;
      }
    }
    
    return false;
  }

  // Encrypt the database file
  async encryptData(data) {
    if (!this.isEncryptionEnabled()) {
      console.log('Encryption key not provided or too short. Skipping encryption.');
      return data;
    }

    // If data is already encrypted, don't re-encrypt it
    if (this.isEncryptedData(data)) {
      console.log('Data appears to be already encrypted. Skipping encryption.');
      return data;
    }

    try {
      // Generate a random initialization vector
      const iv = crypto.randomBytes(16);
      
      // Create cipher with AES-256-CBC using the key and iv
      const key = crypto.createHash('sha256').update(this.encryptKey).digest();
      const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
      
      // Encrypt the data
      const encrypted = Buffer.concat([
        cipher.update(data),
        cipher.final()
      ]);
      
      // Prepend the IV to the encrypted data
      const result = Buffer.concat([iv, encrypted]);
      
      console.log('Data successfully encrypted');
      return result;
    } catch (error) {
      console.error('Error encrypting data:', error.message);
      return data; // Return original data on error
    }
  }

  // Decrypt the database file
  async decryptData(data) {
    if (!this.isEncryptionEnabled()) {
      console.log('Encryption key not provided or too short. Skipping decryption.');
      return data;
    }

    // If data doesn't appear to be encrypted, don't try to decrypt it
    if (!this.isEncryptedData(data)) {
      console.log('Data appears to be in plain text. Skipping decryption.');
      return data;
    }

    try {
      // Extract the IV from the first 16 bytes
      const iv = data.slice(0, 16);
      const encryptedData = data.slice(16);
      
      // Create decipher with AES-256-CBC using the key and iv
      const key = crypto.createHash('sha256').update(this.encryptKey).digest();
      const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
      
      // Decrypt the data
      const decrypted = Buffer.concat([
        decipher.update(encryptedData),
        decipher.final()
      ]);
      
      console.log('Data successfully decrypted');
      return decrypted;
    } catch (error) {
      console.error('Error decrypting data:', error.message);
      return data; // Return original data on error
    }
  }

  // Download database from GitHub and overwrite local file
  async downloadDatabase() {
    if (!this.isConfigured()) {
      console.log('GitHub sync not configured. Skipping download.');
      return false;
    }

    try {
      console.log(`Attempting to download database from GitHub repository: ${this.repoName}`);
      
      // Get the content of the database file from GitHub
      // First, try to get the file info to check if it exists
      try {
        const { data } = await this.octokit.repos.getContent({
          owner: this.owner,
          repo: this.repo,
          path: 'database.db',
        });

        // If the file exists, download the binary content
        if (data && data.download_url) {
          const response = await fetch(data.download_url);
          
          if (!response.ok) {
            throw new Error(`Failed to download file: ${response.statusText}`);
          }
          
          // Get the file as ArrayBuffer
          const arrayBuffer = await response.arrayBuffer();
          let buffer = Buffer.from(arrayBuffer);
          
          // Check if the data appears to be encrypted
          const isEncrypted = this.isEncryptedData(buffer);
          
          // Decrypt the data if encryption is enabled and data appears encrypted
          if (this.isEncryptionEnabled() && isEncrypted) {
            console.log('Downloaded database file is encrypted, decrypting...');
            try {
              buffer = await this.decryptData(buffer);
            } catch (decryptError) {
              console.error('Failed to decrypt database:', decryptError.message);
              console.log('Using the encrypted version as fallback');
            }
          } else if (this.isEncryptionEnabled() && !isEncrypted) {
            console.log('Downloaded database file is plaintext, skipping decryption (plaintext to encrypted transition phase)');
          } else {
            console.log('Downloaded database file is plaintext');
          }
          
          // Write the file to local path
          await fs.writeFile(this.dbPath, buffer);
          console.log('Database successfully downloaded and saved locally');
          this.initialSyncCompleted = true;
          return true;
        }
      } catch (error) {
        // File doesn't exist or other error
        if (error.status === 404) {
          console.log('Database file not found on GitHub. This appears to be the first run.');
          console.log('Marking initial sync as completed to allow future uploads.');
          this.initialSyncCompleted = true;
        } else {
          console.error('Error checking database file on GitHub:', error.message);
        }
        return false;
      }
    } catch (error) {
      console.error('Error downloading database from GitHub:', error.message);
      return false;
    }
  }

  // Schedule a GitHub sync
  scheduleSync() {
    // If a sync is already scheduled, just mark as pending (to avoid multiple timers)
    if (this.syncTimer) {
      console.log('Sync already scheduled. Marking as pending.');
      this.pendingSync = true;
      return; // No need to return a promise here, scheduling is synchronous
    }

    // Otherwise, schedule a new sync
    console.log(`Scheduling GitHub sync with ${this.syncDelay / 1000} second delay`);
    this.pendingSync = true;

    this.syncTimer = setTimeout(async () => {
      // Reset flags before starting the upload
      this.pendingSync = false;
      this.syncTimer = null;

      console.log('Starting GitHub sync...');
      try {
        await this.uploadDatabase();
        console.log('GitHub sync completed successfully');
      } catch (error) {
        console.error('Error during GitHub sync:', error.message);
      }
    }, this.syncDelay);

    // Return immediately after scheduling
    return Promise.resolve(true);
  }

  // Upload database to GitHub
  async uploadDatabase() {
    if (!this.isConfigured()) {
      console.log('GitHub sync not configured. Skipping upload.');
      return false;
    }

    if (!this.initialSyncCompleted) {
      console.log('Initial sync not completed. Skipping upload to prevent overwriting remote data.');
      return false;
    }

    try {
      console.log(`Uploading database to GitHub repository: ${this.repoName}`);
      
      // Read the local database file
      let content = await fs.readFile(this.dbPath);
      
      // Encrypt the data if encryption is enabled
      if (this.isEncryptionEnabled()) {
        // Only encrypt if not already encrypted
        if (!this.isEncryptedData(content)) {
          console.log('Encrypting database before upload...');
          try {
            content = await this.encryptData(content);
          } catch (encryptError) {
            console.error('Failed to encrypt database:', encryptError.message);
            console.log('Using the unencrypted version as fallback');
          }
        } else {
          console.log('Data is already encrypted, skipping re-encryption');
        }
      }
      
      // Convert to base64
      const contentEncoded = content.toString('base64');
      
      // Try to get the file SHA if it exists (needed for update)
      let fileSha;
      try {
        const { data } = await this.octokit.repos.getContent({
          owner: this.owner,
          repo: this.repo,
          path: 'database.db',
        });
        fileSha = data.sha;
      } catch (error) {
        // File doesn't exist yet, which is fine
      }
      
      // Create or update the file on GitHub
      const result = await this.octokit.repos.createOrUpdateFileContents({
        owner: this.owner,
        repo: this.repo,
        path: 'database.db',
        message: 'Update database',
        content: contentEncoded,
        sha: fileSha, // If undefined, GitHub will create a new file
      });
      
      console.log('Database successfully uploaded to GitHub');
      return true;
    } catch (error) {
      console.error('Error uploading database to GitHub:', error.message);
      return false;
    }
  }
}

module.exports = GitHubSync;
