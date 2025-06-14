# Cloudflare Worker 部署指南

你可以选择以下任一方式进行部署：

## 方式一：快速部署

1. 点击上方的 [![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/dreamhartley/gemini-proxy-panel) 按钮。
2. 根据 Cloudflare 的提示完成部署流程，授权访问你的 GitHub 仓库。
3. 部署完成后，你需要进行必要的 **部署后配置**。
   * 注意，当前项目还在开发中，使用Deploy to Cloudflare一键部署可能导致无法支持后续更新。
   * Worker 版本不支持使用 GitHub 同步数据功能

## 方式二：通过 GitHub Actions 自动部署 (推荐)

这种方式适合希望worker自动与原仓库同步更新的用户。

1. **Fork 仓库**:
   
   * 点击本仓库右上角的 "Fork" 按钮，将此仓库 Fork到你自己的 GitHub 账户下。
2. **获取 Cloudflare 信息**:
   
   * **获取 Account ID**:
     * 登录 Cloudflare Dashboard。
     * 在主页右侧栏找到并复制你的 "账户 ID" (Account ID)。或者，导航到 "Workers & Pages"，在概览页面的右侧也能找到它。
   * **获取 API Token**:
     * 在 Cloudflare Dashboard，点击右上角用户图标 -> "我的个人资料" (My Profile) -> "API 令牌" (API Tokens)。
     * 点击 "创建令牌" (Create Token)。
     * 找到 "Cloudflare Workers 编辑" (Edit Cloudflare Workers) 模板，点击 "使用模板" (Use template)。
     * (可选) 你可以根据需要调整权限范围，但模板默认权限通常足够。确保至少包含 `Account` 资源的 `Workers Scripts:Edit` 和 `Workers KV Storage:Edit` 权限。
     * 选择你的账户资源和区域资源（通常保持默认的 "包括" -> "所有区域"）。
     * 点击 "继续以显示摘要" (Continue to summary)，然后点击 "创建令牌" (Create Token)。
     * **立即复制生成的 API Token**，这个 Token 只会显示一次，请妥善保管。
3. **在 GitHub 仓库中设置 Secrets**:
   
   * 进入你 **Fork 后的仓库** 的 GitHub 页面。
   * 点击 "Settings" -> "Secrets and variables" -> "Actions"。
   * 点击 "New repository secret" 按钮，添加以下两个 Secrets：
     * `CF_ACCOUNT_ID`: 粘贴你之前获取的 Cloudflare Account ID。
     * `CF_API_TOKEN`: 粘贴你之前创建并复制的 Cloudflare API Token。
4. **(可选) 配置 PAT 以自动更新 Actions**:
   
   * 如果你希望 GitHub Actions 在运行时能够自动更新工作流文件（例如，当上游仓库更新了 `.github/workflows/deploy.yml` 文件后，你的 Fork 仓库也能通过某种机制自动拉取更新），或者执行需要写权限的操作，你可能需要配置一个 Personal Access Token (PAT)。
   * **创建 PAT**:
     * 前往你的 GitHub "Settings" -> "Developer settings" -> "Personal access tokens" -> "Tokens (classic)"。
     * 点击 "Generate new token" -> "Generate new token (classic)"。
     * 给 Token 起一个描述性的名字，例如 `WORKFLOW_UPDATE_PAT`。
     * 设置过期时间。
     * 在 "Select scopes" 中，勾选 `workflow` (Update GitHub Action workflows)。
     * 点击 "Generate token"。
     * **立即复制生成的 PAT**，并妥善保管。
   * **设置 PAT Secret**:
     * 回到你 Fork 仓库的 "Settings" -> "Secrets and variables" -> "Actions"。
     * 点击 "New repository secret"，添加以下 Secret：
       * `PAT`: 粘贴你刚刚创建并复制的 GitHub Personal Access Token。
5. **触发部署**:
   
   * 当你向 Fork 仓库的 `main` 分支推送 (push) 代码更改时，GitHub Actions 会自动触发部署流程。
   * 你也可以在仓库的 "Actions" 标签页中，找到 "Deploy to Cloudflare Workers" 工作流，手动触发运行。
6. **完成部署后配置**:
   
   * 首次通过 GitHub Actions 部署成功后，你仍然需要按照下面的 **部署后配置** 部分，在 Cloudflare Dashboard 中创建和绑定 KV Namespace，并设置环境变量。

## 方式三：手动部署

1. 克隆此仓库:
   ```bash
   git clone https://github.com/dreamhartley/gemini-proxy-panel.git
   cd gemini-proxy-panel
   ```
2. 安装依赖:
   ```bash
   npm install
   ```
3. 登录 Wrangler:
   ```bash
   npx wrangler login
   ```
4. (可选) 修改 `wrangler.toml` 文件中的 `name`（Worker 名称）和 `account_id`。如果你通过 GitHub Actions 部署，`account_id` 会从 Secret 读取。
5. 部署:
   ```bash
   npx wrangler deploy
   ```

## 部署后配置

在首次部署成功后，需要在 Cloudflare Dashboard 中执行以下配置：

1. **创建 KV Namespace**: **（自动部署无须处理）**
   
   * 在 Cloudflare Dashboard 中，导航到 "Workers & Pages" -> "KV"。
   * 创建两个 KV Namespace:
     * `GEMINI_KEYS_KV`
     * `WORKER_CONFIG_KV`
   * 记下它们的 Namespace ID。
2. **将 KV Namespace 绑定到 Worker**: **（自动部署无须处理）**
   
   * 导航到你部署的 Worker (位于 "Workers & Pages" 下)。
   * 进入 Worker 的 "设置" (Settings) -> "变量" (Variables)。
   * 在 "KV Namespace 绑定" (KV Namespace Bindings) 部分，点击 "编辑变量" (Edit variables)，然后添加两个绑定:
     * 变量名称 (Variable name): `GEMINI_KEYS_KV`, KV Namespace: 选择你刚创建的 `GEMINI_KEYS_KV`。
     * 变量名称 (Variable name): `WORKER_CONFIG_KV`, KV Namespace: 选择你刚创建的 `WORKER_CONFIG_KV`。
   * 点击 "保存" (Save)。
3. **设置环境变量**: **（都需要处理）**
   
   * 在同一个 Worker 的 "设置" (Settings) -> "变量" (Variables) 页面。
   * 在 "环境变量" (Environment Variables) 部分，点击 "编辑变量" (Edit variables)，然后添加以下变量 (类型选择**密钥**):
     * `ADMIN_PASSWORD`: 设置一个安全的密码，用于登录管理面板。
     * `SESSION_SECRET_KEY`: 设置一个长且随机的字符串，用于会话管理。你可以使用密码生成器生成一个强随机字符串（例如，至少 32 个字符）。
   * 点击 "保存" (Save)。
4. **重新部署 (如果需要)**:
   
   * Cloudflare 通常会自动应用绑定和环境变量的更改。但如果 Worker 没有立即获取到最新的绑定和变量，你可能需要手动触发一次新的部署（例如，通过 Wrangler `npx wrangler deploy`，或者在 Cloudflare Dashboard 编辑代码后点击 "部署"，或者通过 GitHub Actions 再次推送或手动触发）。