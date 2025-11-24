import requests

def ddg_search(query):
    """Perform DuckDuckGo search without API keys."""
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }


        resp = requests.get(url, params=params).json()


        if "RelatedTopics" in resp:
            items = resp["RelatedTopics"]
            results = []


            for item in items:
                if isinstance(item, dict) and "Text" in item:
                    results.append(f"- {item['Text']}")
                if len(results) == 3:
                    break


            return "\n".join(results) if results else "No relevant results."


        return "No web search results found."


    except Exception as e:
        return f"Web search error: {str(e)}"