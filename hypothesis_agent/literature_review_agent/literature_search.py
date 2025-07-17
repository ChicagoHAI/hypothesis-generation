import os
import requests
from tqdm import tqdm
from hypogenic.logger_config import LoggerConfig

def auto_literature_search(
    topic: str,
    num_papers: int,
    task_name: str = None,
    num_papers_per_trial: int = 10,
    max_trial: int = 5,
):
    if task_name is None:
        logger.warning(f"need to specify task name")
        return ""
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), f"literature/{task_name}/raw"
    )
    os.makedirs(save_dir, exist_ok=True)
    logger = LoggerConfig.get_logger("auto-literature-search")

    logger.info(f"searching for papers on topic {topic}")

    params = {
        "query": topic,
        "limit": num_papers_per_trial,
        "fields": "title,openAccessPdf",
        "offset": 0,
    }

    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    headers = {
        "x-api-key": os.getenv("SS_API_KEY"),
    }

    papers_list = []
    cnt_papers = 0
    for n_trial in range(max_trial):
        if cnt_papers >= num_papers:
            break
        params["offset"] = n_trial * num_papers_per_trial
        response = requests.get(API_URL, params=params, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Error when querying Semantic Scholar API. Message: {response.text}")
            return ""

        papers = response.json()["data"]
        logger.info(f"found {len(papers)} papers for trial #{n_trial}, starting download")

        for idx, paper in enumerate(tqdm(papers)):
            if cnt_papers >= num_papers:
                break
            title = paper.get("title", f"paper_{idx}")
            logger.info(f"found paper with title {title}")
            pdf_info = paper.get("openAccessPdf")
            if pdf_info and pdf_info.get("url"):
                pdf_url = pdf_info["url"]
                try:
                    pdf_response = requests.get(pdf_url, timeout=10, stream=True)
                    if pdf_response.status_code == 200:
                        content_type = pdf_response.headers.get('Content-Type', '').lower()
                        content_length = int(pdf_response.headers.get('Content-Length', '0'))
                        
                        if 'pdf' in content_type and content_length > 10 * 1024:  # at least 10 KB
                            safe_title = "".join(c if c.isalnum() or c in " ._-" else "_" for c in title)
                            filename = safe_title.strip().replace(" ", "_")[:80] + ".pdf"
                            filepath = os.path.join(save_dir, filename)
                            papers_list.append(filename)
                            
                            with open(filepath, "wb") as f:
                                for chunk in pdf_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            cnt_papers += 1
                            logger.info(f"Successfully downloaded PDF for paper {title}")
                        else:
                            logger.info(f"Skipped (not a valid or too-small PDF): {pdf_url} (size: {content_length} bytes)")
                    else:
                        logger.info(f"Failed to download {title} (HTTP {pdf_response.status_code})")
                except Exception as e:
                    logger.info(f"Error downloading {title}: {e}")
            else:
                logger.info(f"No open access PDF available for '{title}'.")

    logger.info(f"Completed automated literature search, downloaded {cnt_papers} papers:")
    for paper in papers_list:
        logger.info(paper)
    return save_dir
