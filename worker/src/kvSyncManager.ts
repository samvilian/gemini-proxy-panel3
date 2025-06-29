/**
 * KVSyncManager has been refactored to be a simple, stateless wrapper around the Cloudflare KV API.
 * The previous implementation used an in-memory, write-behind cache which introduced significant
 * risks of data loss and stale reads in a distributed worker environment.
 *
 * This new implementation ensures data durability and consistency by writing directly to and
 * reading directly from the KV store for every operation. It no longer holds any state.
 *
 * The singleton pattern is kept to maintain a consistent interface throughout the application,
 * even though the class is now stateless.
 */
export class KVSyncManager {
  private static instance: KVSyncManager;

  // Singleton pattern
  public static getInstance(): KVSyncManager {
    if (!KVSyncManager.instance) {
      KVSyncManager.instance = new KVSyncManager();
    }
    return KVSyncManager.instance;
  }

  private constructor() {
    console.log('KVManager initialized (stateless direct-to-KV mode)');
  }

  /**
   * Writes a value directly to the specified KV namespace.
   * @param namespace The KVNamespace to write to.
   * @param key The key to write.
   * @param value The value to write. It will be JSON.stringified if it is not a string.
   */
  public async setKV(namespace: KVNamespace, key: string, value: any): Promise<void> {
    const valueToStore = typeof value === 'string' ? value : JSON.stringify(value);
    await namespace.put(key, valueToStore);
    // console.log(`[KVManager] Directly wrote to KV: ${key}`);
  }

  /**
   * Reads a value directly from the specified KV namespace.
   * @param namespace The KVNamespace to read from.
   * @param key The key to read.
   * @param type The expected type of the value ('text', 'json', etc.).
   * @returns The value from KV, parsed according to the type.
   */
  public async getKV(namespace: KVNamespace, key: string, type: "text" | "json" | "arrayBuffer" | "stream" = "text"): Promise<any> {
    // console.log(`[KVManager] Directly reading from KV: ${key}`);
    return await namespace.get(key, type as any);
  }

  /**
   * Deletes a key-value pair directly from the specified KV namespace.
   * @param namespace The KVNamespace to delete from.
   * @param key The key to delete.
   */
  public async deleteKV(namespace: KVNamespace, key: string): Promise<void> {
    await namespace.delete(key);
    // console.log(`[KVManager] Directly deleted from KV: ${key}`);
  }

  /**
   * Lists keys directly from the specified KV namespace.
   * @param namespace The KVNamespace to list from.
   * @param options The list options.
   * @returns A promise that resolves to the list result.
   */
  public async listKV(namespace: KVNamespace, options?: KVNamespaceListOptions): Promise<KVNamespaceListResult<unknown>> {
    return await namespace.list(options);
  }

  // The following methods are no longer needed in the direct-to-KV model
  // but are kept as stubs to prevent breaking changes if called somewhere.
  public async forceSyncAll(): Promise<void> {
    // console.log('[KVManager] forceSyncAll is a no-op in direct-to-KV mode.');
    return Promise.resolve();
  }

  public registerNamespace(namespace: KVNamespace): void {
    // console.log('[KVManager] registerNamespace is a no-op in direct-to-KV mode.');
  }
}

// Export the singleton instance
export const kvSyncManager = KVSyncManager.getInstance();
