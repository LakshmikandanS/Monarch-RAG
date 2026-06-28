import time
import random
from datetime import datetime
import re
from pathlib import PurePosixPath
from urllib.parse import urlparse, urljoin, urldefrag
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from curl_cffi import requests as curl_requests
from curl_cffi.requests import errors as curl_exceptions
from curl_cffi.requests.exceptions import RequestException, Timeout

# Suppress the BeautifulSoup XML warning globally
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def create_session():
    session = curl_requests.Session(
        impersonate="chrome120",  # Spoofs Chrome's exact TLS Cipher suite
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    return session

def get_soup(url, session, referer="https://www.google.com/", max_retries=3):
    """
    Fetches the HTML soup with automatic exponential backoff retries to handle Timeouts.
    """
    for attempt in range(max_retries):
        sleep_time = random.uniform(1.5, 4.0)
        time.sleep(sleep_time)
        
        try:
            response = session.get(url, timeout=15, headers={"Referer": referer})
            response.raise_for_status()
            
            # UPGRADE: Skip non-HTML content (like XML sitemaps, RSS feeds, or SVGs)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
                print(f"  [Info] Skipping non-HTML content at {url} (Type: {content_type})")
                return None
            
            res_lower = response.text.lower()
            if "cf-turnstile" in res_lower or "just a moment" in res_lower:
                raise Exception("Encountered Cloudflare Turnstile challenge.")
                
            return BeautifulSoup(response.text, 'html.parser')
            
        except (Timeout, RequestException) as e:
            print(f"  [Warning] Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to fetch {url} after {max_retries} attempts.")
            time.sleep(2 ** attempt)  # Exponential backoff: sleep 1s, 2s, 4s...
        except Exception as e:
            raise Exception(f"An error occurred while fetching {url}: {e}")

def extract_title(soup):
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else None

def extract_links(soup, base_url):
    unique_links = set()
    seed_domain = urlparse(base_url).netloc
    root_word = base_url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
    
    for link in soup.find_all('a'):
        href = link.get('href')
        if not href:
            continue
        full_url = urljoin(base_url, href)
        full_url, _ = urldefrag(full_url) # Remove #anchor links to prevent duplicate crawls
        
        # Keep crawling within the same documentation bounds
        if full_url.count(root_word) > 2:
            continue
            
        parsed = urlparse(full_url)
        if parsed.netloc == seed_domain:
            unique_links.add(full_url)
            
    return list(unique_links)

def extract_sections(soup):
    """
    Parses HTML into clean Markdown, normalizing whitespace to keep Regex safe.
    UPGRADE: Added Support for Lists (li) and better Sphinx code block formatting.
    """
    root = soup.find("main") or soup.find("article") or soup.find("body") or soup

    for noise in root.find_all(
        ["script", "style", "nav", "footer", "header", "aside"]
    ):
        noise.decompose()

    formatted_text = []

    # UPGRADE: Added 'li' to catch bullet points (heavily used in D2L summaries)
    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "li"]):
        # Use separator=" " to prevent math variables from mashing together
        text = element.get_text(separator=" ", strip=True) 
        if not text:
            continue

        tag = element.name.lower()

        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # CRITICAL: Squash internal rogue newlines inside headers
            clean_header = re.sub(r"\s+", " ", text)
            level = tag[1]
            formatted_text.append(f"{'#' * int(level)} {clean_header}")

        elif tag == "pre":
            # Clean up Sphinx copy-button artifacts if present
            code_content = element.get_text()
            formatted_text.append(f"```\n{code_content.strip()}\n```")
            
        elif tag == "li":
            # Convert list items to markdown bullets
            clean_li = re.sub(r"\s+", " ", text)
            formatted_text.append(f"* {clean_li}")
            
        elif tag == "p":
            clean_p = re.sub(r"\s+", " ", text)
            formatted_text.append(clean_p)

    return "\n\n".join(formatted_text).strip()


def get_sections_with_hierarchy(content):
    """
    Extracts hierarchy, character spans, AND the actual payload content.
    """
    cleaned_content = content.replace("\r\n", "\n")
    header_pattern = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)

    matches = list(header_pattern.finditer(cleaned_content))

    sections = []
    hierarchy_stack = []

    # 1. Capture "Preface" text that appears before the very first header
    first_header_index = matches[0].start() if matches else len(cleaned_content)
    if first_header_index > 0:
        preface = cleaned_content[:first_header_index].strip()
        if preface:
            sections.append(
                {
                    "section_path": "Preface",
                    "header": None,
                    "level": 0,
                    "start_char": 0,
                    "end_char": first_header_index,
                    "content": preface,
                }
            )

    # 2. Iterate headers and bind their text bodies
    for i, match in enumerate(matches):
        level = len(match.group(1))
        header_text = match.group(2).strip()
        start_char = match.start()

        # The section ends where the next header begins (or at the EOF)
        end_char = (
            matches[i + 1].start() if (i + 1) < len(matches) else len(cleaned_content)
        )

        # Maintain hierarchy stack
        while hierarchy_stack and hierarchy_stack[-1][0] >= level:
            hierarchy_stack.pop()

        hierarchy_stack.append((level, header_text))
        hierarchy_path = " > ".join([h[1] for h in hierarchy_stack])

        # Slice the document from the end of the `# Header` line to the start of the next one
        body_content = cleaned_content[match.end() : end_char].strip()

        sections.append(
            {
                "section_path": hierarchy_path,
                "header": header_text,
                "level": level,
                "start_char": start_char,
                "end_char": end_char,
                "content": body_content,
            }
        )

    return sections

def derive_file_name(url, title=None):
    parsed_url = urlparse(url)
    path_name = PurePosixPath(parsed_url.path.rstrip("/")).name

    if path_name:
        return path_name

    if title:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
        if slug:
            return f"{slug}.html"

    domain = parsed_url.netloc.replace(":", "_")
    return f"{domain or 'scraped_document'}.html"

def create_document(url, active_session, current_referer):
    soup = get_soup(url, active_session, referer=current_referer)
    if soup is None:
        raise Exception(f"Failed to retrieve or parse the content from {url}.")
    
    title = extract_title(soup)
    links = extract_links(soup, url)
    content = extract_sections(soup)
    file_name = derive_file_name(url, title)
    
    # Calculate sections hierarchy immediately using the extracted content
    hierarchy_sections = get_sections_with_hierarchy(content)

    document = {
        'url': url,
        'title': title,
        'links': links,
        'content': content,
        'sections': hierarchy_sections,  # Attached hierarchy directly to document
        'metadata': {
            'file_name': file_name,
            'source': url,
            'url': url,
            'title': title,
            'domain': urlparse(url).netloc,
            'referer': current_referer,
            'created_at': datetime.now().isoformat(),
        }
    }
    return document

def crawler(start_url, session, initial_referer="https://www.google.com/"):
    visited = set()
    queued = {start_url}
    to_visit = [(start_url, initial_referer)]
    documents = []
    
    while to_visit:
        current_url, referer = to_visit.pop(0)
        
        if current_url in visited:
            continue
            
        print(f"Crawling: {current_url}")
        try:
            document = create_document(current_url, session, referer)
            documents.append(document)
            visited.add(current_url)
            
            for link in document['links']:
                if link not in visited and link not in queued:
                    queued.add(link)
                    to_visit.append((link, current_url))
                    
        except Exception as e:
            print(f"  [Error] Skipping {current_url}: {e}")
            visited.add(current_url) 
    
    return documents

# Example Usage:
# if __name__ == "__main__":
#     sess = create_session()
#     docs = crawler("https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html", sess)
#     print(f"Scraped {len(docs)} documents.")