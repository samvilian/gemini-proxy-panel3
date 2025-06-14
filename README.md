# Gemini Proxy Panel

> **本项目遵循CC BY-NC 4.0协议，禁止任何形式的商业倒卖行为。**  
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
Commercial resale or any form of commercial use is prohibited.

[**中文介绍**](./README_zh.md "Chinese Readme") <br><br>
[***详细部署与使用文档(新手看这里)***](./doc/项目介绍.md "项目介绍") <br><br>
[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/dreamhartley/gemini-proxy-panel)

## Introduction

`gemini-proxy-panel` is a proxy service deployed on Cloudflare Workers. It forwards requests formatted for the OpenAI API to the Google Gemini Pro API, allowing applications developed for OpenAI to seamlessly switch to or leverage the capabilities of Gemini models.

## Features

*   **OpenAI to Gemini Proxy**: Seamlessly translates OpenAI Chat API requests into Gemini Pro API requests.
*   **Multi-API Key Rotation**: Supports configuring multiple Gemini API keys and automatically rotates through them to distribute request load and circumvent rate limits.
*   **Quota and Usage Management**: Monitor the usage of each Gemini API key through an intuitive management interface.
*   **Key Management**: Centrally manage multiple Gemini API keys and Worker API keys (used to access this proxy service) within the management panel.
*   **Model Configuration**: Define and manage the Gemini models supported by this proxy in the management panel.
*   **Intuitive Management Interface**: Provides a Web UI (`/login` or `/admin`) to view API usage statistics and configure settings.
*   **One-Click Deployment**: Supports quick deployment to the Cloudflare Workers platform via the "Deploy to Cloudflare" button.
*   **GitHub Actions Automatic Deployment**: After forking the repository, enables automatic deployment via GitHub Actions upon code push.
*   **GitHub Database Sync**: Leverages GitHub repositories for automatic database synchronization.

## Hugging Face Space Deployment

This deployment method utilizes Hugging Face Space's Docker environment and **requires enabling GitHub sync** for data persistence.

1. **Prepare GitHub Repository and PAT**:
   
   * You need **your own** GitHub repository to store synchronized data. A private repository is recommended.
   * Create a GitHub Personal Access Token (PAT) with the `repo` permission scope. **Keep this token secure**.

2. **Create a Hugging Face Space**:
   
   * Visit Hugging Face and create a new Space.
   * Select "Docker" as the Space SDK.
   * Choose "Use existing Dockerfile from repository".

3. **Configure Space Secrets**:
   
   * Go to your Space's "Settings" -> "Repository secrets".
   * Add the following Secrets:
     * `ADMIN_PASSWORD`: Set a login password for the admin panel.
     * `SESSION_SECRET_KEY`: Set a long, random session key.
     * `GITHUB_PROJECT`: Enter **your own** GitHub repository path in the format `your-username/your-repo-name`.
     * `GITHUB_PROJECT_PAT`: Enter your GitHub PAT created earlier.
     * `GITHUB_ENCRYPT_KEY`: Set an encryption key for synced data, **must be at least 32 characters long**.

4. **Create Dockerfile**:
   
   * In your Hugging Face Space's "Files" tab, click "Add file" -> "Create new file".
   * Set the filename to `Dockerfile`.
   * Paste the following content into the file:
     ```dockerfile
     FROM dreamhartley705/gemini-proxy-panel:huggingface
     ```
   * Click "Commit new file".

5. **Launch and Access**:
   
   * Hugging Face Space will automatically build and start the application using this `Dockerfile`.
   * Once launched, the app will connect to your GitHub repository for data syncing using your configured Secrets.
   * You can access the admin panel (`/login` or `/admin`) and API (`/v1`) via the URL provided by the Space.

## Local Node.js Deployment

This method is suitable for local development and testing.

1. **Clone the Repository**:
    
    ```bash
    git clone https://github.com/dreamhartley/gemini-proxy-panel.git
    cd gemini-proxy-panel
    ```

2. **Install Dependencies**:
    
    ```bash
    npm install
    ```

3. **Configure Environment Variables**:
    
    * Copy the `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    * Edit the `.env` file, setting at minimum:
        * `ADMIN_PASSWORD`: Set the admin panel login password.
        * `SESSION_SECRET_KEY`: Set a long, random string for session security (e.g., generate with `openssl rand -base64 32`).
        * `PORT` (optional): Default is 3000, change as needed.
    * **(Optional) Configure GitHub Sync**: To sync data to a GitHub repository, set:
        * `GITHUB_PROJECT`: Your GitHub repository path in format `username/repo-name`. **Note: This is your own repository for data backup, not this project's repository.**
        * `GITHUB_PROJECT_PAT`: Your GitHub Personal Access Token with `repo` permission.
        * `GITHUB_ENCRYPT_KEY`: An encryption key for syncing data, must be at least 32 characters long.

4. **Start the Service**:
    
    ```bash
    npm start
    ```
    
    The service will run at `http://localhost:3000` (or your configured port).

## Docker Deployment

You can quickly deploy using Docker or Docker Compose.

### Method 1: Using `docker build` and `docker run`

1. **Clone the Repository**:
    
    ```bash
    git clone https://github.com/dreamhartley/gemini-proxy-panel.git
    cd gemini-proxy-panel
    ```

2. **Configure Environment Variables**:
    
    * Copy the `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    * Edit the `.env` file, setting the necessary variables (`ADMIN_PASSWORD`, `SESSION_SECRET_KEY`) and optional GitHub sync variables (`GITHUB_PROJECT`, `GITHUB_PROJECT_PAT`, `GITHUB_ENCRYPT_KEY`). **Note: The `PORT` variable typically doesn't need to be set in the `.env` file for Docker deployments, as port mapping is done in the `docker run` command.**

3. **Build the Docker Image**:
    
    ```bash
    docker build -t gemini-proxy-panel .
    ```

4. **Run the Docker Container**:
    
    ```bash
    docker run -d --name gemini-proxy-panel \
      -p 3000:3000 \
      --env-file .env \
      -v ./data:/usr/src/app/data \
      gemini-proxy-panel
    ```
    
    * `-d`: Run the container in the background.
    * `--name gemini-proxy-panel`: Name for the container.
    * `-p 3000:3000`: Map host port 3000 to container port 3000.
    * `--env-file .env`: Load environment variables from the `.env` file.
    * `-v ./data:/usr/src/app/data`: Mount the local `data` directory to the container for SQLite database persistence. Ensure the `data` directory exists locally.

### Method 2: Using `docker-compose` (Recommended)

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/dreamhartley/gemini-proxy-panel.git
   cd gemini-proxy-panel
   ```

2. **Configure Environment Variables**:
   
   * Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   * Edit the `.env` file, setting the necessary variables (`ADMIN_PASSWORD`, `SESSION_SECRET_KEY`) and optional GitHub sync variables (`GITHUB_PROJECT`, `GITHUB_PROJECT_PAT`, `GITHUB_ENCRYPT_KEY`).

3. **Start the Service**:
   
   ```bash
   docker-compose up -d
   ```
   
   Docker Compose will automatically build the image (if needed), create and start the container, and handle port mapping, environment variables, and data volumes according to the `docker-compose.yml` file.

## Cloudflare Worker Deployment

You can choose one of the following methods for deployment:

### Method 1: Quick Deployment

1.  Click the [![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/dreamhartley/gemini-proxy-panel) button above.
2.  Follow the prompts from Cloudflare to complete the deployment process, authorizing access to your GitHub repository.
3.  After deployment, you will need to perform the necessary **Post-Deployment Configuration**.
    *   Note: This project is currently under development. Using the "Deploy to Cloudflare" one-click deployment might prevent receiving future updates.
    *   The Worker version does not support using the GitHub sync data feature.

### Method 2: Automatic Deployment via GitHub Actions(Recommended)

This method is suitable for users who want their worker to automatically sync updates from the original repository.

1.  **Fork the Repository**:
    *   Click the "Fork" button in the upper right corner of this repository to fork it to your own GitHub account.

2.  **Obtain Cloudflare Information**:
    *   **Get Account ID**:
        *   Log in to the Cloudflare Dashboard.
        *   Find and copy your "Account ID" from the right sidebar on the main page. Alternatively, navigate to "Workers & Pages", and you can find it on the overview page's right side.
    *   **Get API Token**:
        *   In the Cloudflare Dashboard, click your user icon in the top right -> "My Profile" -> "API Tokens".
        *   Click "Create Token".
        *   Find the "Edit Cloudflare Workers" template and click "Use template".
        *   (Optional) You can adjust the permission scope as needed, but the default template permissions are usually sufficient. Ensure it includes at least `Workers Scripts:Edit` and `Workers KV Storage:Edit` permissions for the `Account` resource.
        *   Select your account resources and zone resources (usually keep the default "Include" -> "All zones").
        *   Click "Continue to summary", then click "Create Token".
        *   **Immediately copy the generated API Token**. This token is shown only once, so store it securely.

3.  **Set Secrets in Your GitHub Repository**:
    *   Go to the GitHub page of **your forked repository**.
    *   Click "Settings" -> "Secrets and variables" -> "Actions".
    *   Click the "New repository secret" button and add the following two secrets:
        *   `CF_ACCOUNT_ID`: Paste your Cloudflare Account ID obtained earlier.
        *   `CF_API_TOKEN`: Paste the Cloudflare API Token you created and copied earlier.

4.  **(Optional) Configure PAT for Automatic Action Updates**:
    *   If you want GitHub Actions to be able to automatically update workflow files during runtime (e.g., if the upstream repository updates `.github/workflows/deploy.yml`, your fork could potentially pull these updates automatically via some mechanism), or perform other actions requiring write permissions to the repository, you might need to configure a Personal Access Token (PAT).
    *   **Create PAT**:
        *   Go to your GitHub "Settings" -> "Developer settings" -> "Personal access tokens" -> "Tokens (classic)".
        *   Click "Generate new token" -> "Generate new token (classic)".
        *   Give the token a descriptive name, e.g., `WORKFLOW_UPDATE_PAT`.
        *   Set an expiration date.
        *   Under "Select scopes", check the `workflow` scope (Update GitHub Action workflows).
        *   Click "Generate token".
        *   **Immediately copy the generated PAT** and store it securely.
    *   **Set PAT Secret**:
        *   Return to your forked repository's "Settings" -> "Secrets and variables" -> "Actions".
        *   Click "New repository secret" and add the following secret:
            *   `PAT`: Paste the GitHub Personal Access Token you just created and copied.

5.  **Trigger Deployment**:
    *   When you push code changes to the `main` branch of your forked repository, the GitHub Actions deployment workflow will automatically trigger.
    *   You can also go to the "Actions" tab of your repository, find the "Deploy to Cloudflare Workers" workflow, and manually trigger a run.

6.  **Complete Post-Deployment Configuration**:
    *   After the first successful deployment via GitHub Actions, you still need to follow the **Post-Deployment Configuration** steps below to create and bind KV Namespaces and set environment variables in the Cloudflare Dashboard.

### Method 3: Manual Deployment (Using Wrangler)

1.  Clone this repository:
    ```bash
    git clone https://github.com/dreamhartley/gemini-proxy-panel.git
    cd gemini-proxy-panel
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Log in to Wrangler:
    ```bash
    npx wrangler login
    ```
4.  (Optional) Modify the `name` (Worker name) and `account_id` in the `wrangler.toml` file. If deploying via GitHub Actions, `account_id` will be read from the secret.
5.  Deploy:
    ```bash
    npx wrangler deploy
    ```

### Post-Deployment Configuration (Critical Steps)

**Regardless of the deployment method used**, after the initial successful deployment, you need to perform the following configurations in the Cloudflare Dashboard:

1.  **Create KV Namespaces**: **(No need to handle for automatic deployment)**
    *   In the Cloudflare Dashboard, navigate to "Workers & Pages" -> "KV".
    *   Create two KV Namespaces:
        *   `GEMINI_KEYS_KV`
        *   `WORKER_CONFIG_KV`
    *   Note down their Namespace IDs.

2.  **Bind KV Namespaces to Worker**: **(No need to handle for automatic deployment)**
    *   Navigate to your deployed Worker (under "Workers & Pages").
    *   Go to the Worker's "Settings" -> "Variables".
    *   In the "KV Namespace Bindings" section, click "Edit variables", then add two bindings:
        *   Variable name: `GEMINI_KEYS_KV`, KV Namespace: Select the `GEMINI_KEYS_KV` you just created.
        *   Variable name: `WORKER_CONFIG_KV`, KV Namespace: Select the `WORKER_CONFIG_KV` you just created.
    *   Click "Save".

3.  **Set Environment Variables**: **(Required for all methods)**
    *   On the same Worker's "Settings" -> "Variables" page.
    *   In the "Environment Variables" section, click "Edit variables", then add the following variables (Select type **Secret**):
        *   `ADMIN_PASSWORD`: Set a secure password to log in to the management panel.
        *   `SESSION_SECRET_KEY`: Set a long and random string for session management. You can use a password generator to create a strong random string (e.g., at least 32 characters).
    *   Click "Save".

4.  **Redeploy (If Necessary)**:
    *   Cloudflare usually applies binding and environment variable changes automatically. However, if the Worker doesn't immediately pick up the latest bindings and variables, you might need to manually trigger a new deployment (e.g., via Wrangler `npx wrangler deploy`, by editing code in the Cloudflare dashboard and clicking "Deploy", or by pushing again/manually triggering the GitHub Action).

## Usage

### Management Panel

1.  Access the `/login` or `/admin` path of your Worker URL (e.g., `https://your-worker-name.your-subdomain.workers.dev/login`).
2.  Log in using the `ADMIN_PASSWORD` you set.
3.  In the management panel, you can:
    *   Add and manage your Gemini API keys.
    *   Add and manage API keys used to access this Worker proxy (Worker API Keys).
    *   Set global quotas for Pro and Flash series models.
    *   View usage statistics for each Gemini API key.
    *   Configure supported Gemini models.

### API Proxy

1.  Point the API endpoint of your application (originally configured to call the OpenAI API) to your deployed Worker URL (e.g., `https://your-worker-name.your-subdomain.workers.dev/v1`).
2.  Ensure that your application includes valid authentication information when sending requests. This is usually done by carrying the "Worker API Key" configured in the management panel in the `Authorization` request header:
    ```
    Authorization: Bearer <your_worker_api_key>
    ```
3.  Send requests compatible with the OpenAI Chat Completions API. The Worker will convert them into Gemini API requests and return the formatted response.

## Configuration Overview

## Configuration Overview

### Local Node.js / Docker / Hugging Face Deployments

These deployment methods configure environment variables through the `.env` file or Secrets (Hugging Face).

* **Core Environment Variables (Required)**:
  * `ADMIN_PASSWORD`: Login password for the admin panel.
  * `SESSION_SECRET_KEY`: Key for securing user sessions (use a long, random string).
* **Optional Environment Variables**:
  * `PORT`: (Local Node.js/Docker only) Port for the service to listen on, default is 3000. Hugging Face handles the port automatically.
* **GitHub Sync Environment Variables (Optional, Required for Hugging Face)**:
  * `GITHUB_PROJECT`: Path to **your own** GitHub repository for data syncing (format: `username/repo-name`).
  * `GITHUB_PROJECT_PAT`: GitHub Personal Access Token with `repo` permission.
  * `GITHUB_ENCRYPT_KEY`: Key for encrypting synced data (at least 32 characters).

### Cloudflare Worker Deployment

* **KV Namespaces (Must Be Bound)**:
  * `GEMINI_KEYS_KV`: Stores Gemini API keys and their usage.
  * `WORKER_CONFIG_KV`: Stores Worker configurations, such as Worker API keys, supported models, etc.
* **Environment Variables (Must Be Set)**:
  * `ADMIN_PASSWORD`: The login password for the management panel.
  * `SESSION_SECRET_KEY`: The key used to protect user session security.
* **GitHub Actions Secrets (Required for Automatic Deployment)**:
  * `CF_ACCOUNT_ID`: Your Cloudflare Account ID.
  * `CF_API_TOKEN`: Cloudflare API Token used for deploying the Worker.
  * `PAT` (Optional): Personal Access Token for GitHub Actions workflow updates or other operations requiring repository write permissions.

<a href="https://dartnode.com" title="Powered by DartNode - Free VPS for Open Source">
  <img src="https://dartnode.com/branding/DN-Open-Source-sm.png" alt="Powered by DartNode" width="300">
</a>
