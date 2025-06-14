// 内存缓存，用于存储临时数据
interface MemoryCache {
  [namespace: string]: {
    [key: string]: {
      value: any;
      timestamp: number;
    };
  };
}

export class KVSyncManager {
  private static instance: KVSyncManager;
  private memoryCache: MemoryCache = {};
  private lastSyncTime: number = 0; // 上次同步时间戳
  private syncDelay: number = 300000; // 5分钟延迟

  // 单例模式
  public static getInstance(): KVSyncManager {
    if (!KVSyncManager.instance) {
      KVSyncManager.instance = new KVSyncManager();
    }
    return KVSyncManager.instance;
  }

  private constructor() {
    console.log('KV同步管理器初始化，设置延迟时间：' + this.syncDelay / 1000 + '秒');
  }

  // 设置KV值（带延迟写入）
  public async setKV(namespace: KVNamespace, key: string, value: any, inMemoryOnly: boolean = false): Promise<void> {
    // 将值存储在内存缓存中
    const namespaceName = this.getNamespaceName(namespace);
    if (!this.memoryCache[namespaceName]) {
      this.memoryCache[namespaceName] = {};
    }
    
    // 存储值和时间戳
    this.memoryCache[namespaceName][key] = {
      value,
      timestamp: Date.now()
    };
    
    console.log(`[KVSync] 值已存储在内存缓存中: ${namespaceName}/${key}`);
    
    // 如果是仅内存标记，不调度同步
    if (inMemoryOnly) {
      return;
    }
    
    // 调度同步
    await this.scheduleSync();
  }

  // 获取KV值（优先从内存缓存获取）
  public async getKV(namespace: KVNamespace, key: string, type: "text" | "json" | "arrayBuffer" | "stream" = "text"): Promise<any> {
    const namespaceName = this.getNamespaceName(namespace);
    
    // 检查内存缓存中是否有值
    if (this.memoryCache[namespaceName] && this.memoryCache[namespaceName][key]) {
      console.log(`[KVSync] 从内存缓存中获取值: ${namespaceName}/${key}`);
      return this.memoryCache[namespaceName][key].value;
    }
    
    // 如果内存缓存没有，从KV存储获取
    console.log(`[KVSync] 从KV存储获取值: ${namespaceName}/${key}`);
    const value = await namespace.get(key, type as any);
    
    // 将获取的值存储在内存缓存中
    if (value !== null) {
      if (!this.memoryCache[namespaceName]) {
        this.memoryCache[namespaceName] = {};
      }
      this.memoryCache[namespaceName][key] = {
        value,
        timestamp: Date.now()
      };
    }
    
    return value;
  }

  // 删除KV值
  public async deleteKV(namespace: KVNamespace, key: string): Promise<void> {
    const namespaceName = this.getNamespaceName(namespace);
    
    // 从内存缓存中删除
    if (this.memoryCache[namespaceName] && this.memoryCache[namespaceName][key]) {
      delete this.memoryCache[namespaceName][key];
      console.log(`[KVSync] 从内存缓存中删除值: ${namespaceName}/${key}`);
    }
    
    // 从KV存储中删除
    await namespace.delete(key);
    console.log(`[KVSync] 从KV存储中删除值: ${namespaceName}/${key}`);
  }

  // 列出命名空间中的所有键
  public async listKV(namespace: KVNamespace, options?: KVNamespaceListOptions): Promise<KVNamespaceListResult<unknown>> {
    return await namespace.list(options);
  }

  // 强制执行同步，用于定时任务或请求结束前确保数据同步
  public async forceSyncAll(): Promise<void> {
    console.log('[KVSync] 执行强制同步');
    await this.syncKVs();
    this.lastSyncTime = Date.now();
    console.log('[KVSync] 强制同步完成，更新同步时间戳:', this.lastSyncTime);
  }

  // 调度同步操作 - 基于时间戳检查
  private async scheduleSync(): Promise<boolean> {
    const currentTime = Date.now();
    
    // 检查是否已经超过设定的延迟时间(syncDelay)
    if (currentTime - this.lastSyncTime < this.syncDelay) {
      // 如果未达到同步时间间隔，记录日志并返回
      console.log(`[KVSync] 距离上次同步未满${this.syncDelay / 1000}秒，跳过本次同步`);
      return true;
    }
    
    // 已达到同步时间，立即执行同步
    console.log('[KVSync] 开始KV同步...');
    try {
      await this.syncKVs();
      // 更新最后同步时间戳
      this.lastSyncTime = Date.now();
      console.log('[KVSync] KV同步完成，更新同步时间戳:', this.lastSyncTime);
    } catch (error) {
      console.error('[KVSync] KV同步过程中出错:', error);
    }
    
    return true;
  }

  // 同步所有KV
  private async syncKVs(): Promise<void> {
    for (const namespaceName in this.memoryCache) {
      for (const key in this.memoryCache[namespaceName]) {
        try {
          const entry = this.memoryCache[namespaceName][key];
          const namespace = this.getNamespaceFromName(namespaceName);
          
          if (namespace) {
            await namespace.put(key, typeof entry.value === 'string' ? entry.value : JSON.stringify(entry.value));
            console.log(`[KVSync] 已同步值到KV存储: ${namespaceName}/${key}`);
          } else {
            console.error(`[KVSync] 无法找到命名空间: ${namespaceName}`);
          }
        } catch (error) {
          console.error(`[KVSync] 同步值时出错 ${namespaceName}/${key}:`, error);
        }
      }
    }
  }

  // 获取命名空间名称
  private getNamespaceName(namespace: KVNamespace): string {
    // 尝试获取绑定名称，这是确定性的而非随机生成的
    // Worker环境中，每个KV命名空间都有对应的绑定名（如"GEMINI_KEYS_KV"）
    if ((namespace as any).__BINDING_NAME) {
      return (namespace as any).__BINDING_NAME;
    }
    
    // 尝试从注册表中查找已存在的命名空间
    for (const [name, ns] of Object.entries(this.namespaceRegistry)) {
      if (ns === namespace) {
        return name;
      }
    }
    
    // 如果还未注册，返回一个固定前缀的名称
    // 通过使用字符串哈希而非随机数，确保同一命名空间实例在不同请求间有相同标识符
    return `KV_NS_${this.objectToHashString(namespace)}`;
  }
  
  // 将对象转换为哈希字符串，用于生成确定性标识符
  private objectToHashString(obj: any): string {
    // 简单的对象哈希算法，对同一对象生成相同字符串
    const str = String(obj);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash; // 转换为32位整数
    }
    return Math.abs(hash).toString(36);
  }

  // 从名称获取命名空间（需要在使用时注入实际的命名空间）
  private namespaceRegistry: {[name: string]: KVNamespace} = {};
  
  public registerNamespace(namespace: KVNamespace): void {
    const name = this.getNamespaceName(namespace);
    this.namespaceRegistry[name] = namespace;
  }
  
  private getNamespaceFromName(name: string): KVNamespace | null {
    return this.namespaceRegistry[name] || null;
  }
}

// 导出单例实例
export const kvSyncManager = KVSyncManager.getInstance();
