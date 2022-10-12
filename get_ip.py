import requests

# url = "https://httpbin.org/ip"
url = "https://www.baidu.com"
ip = "221.5.80.66:3128"
proxies = {"http": "http://" + ip, "https": "http://" + ip}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36",
    "Connection": "close",
}
resp = requests.get(url, headers=headers, proxies=proxies)

print(resp.status_code)
