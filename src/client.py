import requests
import time
import yaml
import os

# 加载配置
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
if not os.path.exists(_config_path):
    _config_path = 'config.yaml'
cfg = yaml.safe_load(open(_config_path, encoding='utf-8'))

# 从配置中获取默认值
DEFAULT_API_KEY = cfg.get("api_keys", {}).get("the_graph", cfg.get("THE_GRAPH_API_KEY", ""))
MAX_RETRIES = cfg.get("network", {}).get("max_retries", 3)
RETRY_DELAY = cfg.get("network", {}).get("retry_delay", 5)
TIMEOUT = cfg.get("network", {}).get("timeout", 60)

PROXIES = {
    'http': cfg.get("proxy", {}).get("http", "http://127.0.0.1:7890"),
    'https': cfg.get("proxy", {}).get("https", "http://127.0.0.1:7890")
}

class GraphClient:

    def __init__(self, url, api_key=None):
        self.url = url
        self.api_key = api_key or DEFAULT_API_KEY

    def query(self, q):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(self.url, headers=headers, json={"query": q}, timeout=TIMEOUT, proxies=PROXIES)

                if r.status_code != 200:
                    print(f"Request failed with status {r.status_code}: {r.text[:200]}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    raise Exception(r.text)
                
                data = r.json()
                if 'errors' in data:
                    print(f"GraphQL errors: {data['errors']}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    raise Exception(str(data['errors']))
                
                print(f"Query returned {len(data.get('data', {}))} keys")
                return r.json()["data"]
            
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"Error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
        return None

    def paged_query(self, template, key):

        all_data = []
        skip = 0
        print(f"Fetching {key} with pagination...")
        
        while True:
            # 使用字符串替换而不是 format，避免大括号冲突
            q = template.replace("{skip}", str(skip))

            result = self.query(q)
            if result is None:
                print(f"Query returned None, stopping pagination")
                break
                
            data = result.get(key, [])
            
            if len(data) == 0:
                print(f"No more data at skip={skip}")
                break

            all_data.extend(data)
            print(f"  Page at skip={skip}: +{len(data)} records (total: {len(all_data)})")
            skip += len(data)

            time.sleep(0.2)

        print(f"Finished fetching {key}: total {len(all_data)} records")
        return all_data
