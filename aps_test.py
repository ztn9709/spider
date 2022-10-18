import random
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from retry import retry
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

mgz_list = ["prb", "prl", "pra", "prresearch", " rmp"]
mgz = "prb"
host = "https://journals.aps.org/"
page_range = range(563, 760)


chrome_path = "C:\\ProgramData\\Anaconda3\\chromedriver.exe"
options = webdriver.ChromeOptions()
options.page_load_strategy = "eager"
options.add_argument("headless")
options.add_argument("--disable-gpu")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--ignore-ssl-errors")
options.add_argument("--ignore-certificate-errors")
options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
UA = "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
options.add_argument(UA)


def random_sleep(mu, sigma):
    secs = random.normalvariate(mu, sigma)
    if secs <= 0:
        secs = mu  # 太小则重置为平均值
    time.sleep(secs)


@retry()
def get_paper_link(url):
    driver = webdriver.Chrome(service=Service(chrome_path), options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "lxml")
    paper_links = []
    for result in soup.select(".article.panel.article-result"):
        link = host + mgz + "/abstract/" + result.attrs["data-id"]
        paper_links.append(link)
    driver.close()
    driver.quit()
    return paper_links


@retry()
def get_details(url):
    driver = webdriver.Chrome(service=Service(chrome_path), options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.quit()
    data = {
        "link": url,
        "title": "",
        "authors": [],
        "institutes": [],
        "DOI": "",
        "abstract": "",
        "date": "",
        "publication": "",
        "areas": [],
        "keywords": [],
        "topo_label": 0,
    }
    data["title"] = soup.select("h3")[0].text
    for author in soup.select(".authors .content p a"):
        if author.text:
            data["authors"].append(author.text)
    if soup.select(".authors .content ul"):
        for institute in soup.select(".authors .content ul")[0].select("li"):
            data["institutes"].append(re.sub(r"^\d", "", institute.text))
    data["DOI"] = soup.select(".doi-field")[0].text
    if soup.select(".abstract .content p"):
        data["abstract"] = soup.select(".abstract .content p")[0].text
    data["date"] = soup.find(attrs={"property": "article:published_time"}).attrs["content"]
    data["publication"] = soup.select("h2")[0].text
    if soup.select(".physh-concepts"):
        for area in soup.select(".physh-concepts")[0].select(".physh-concept"):
            data["areas"].append(area.text)
    data["topo_label"] = is_topo(data)
    return data


def is_topo(data):
    topo_areas = [
        "Topological phases of matter",
        "Symmetry protected topological states",
        "Topological insulators",
        "Topological order",
        "Topological phase transition",
        "Topological superconductors",
        "Topological materials",
    ]
    for area in topo_areas:
        if area in data["areas"]:
            return 1
    return 0


def save2mongodb(data):
    url = "http://localhost:4000/api/paper"
    r = requests.post(url, data=data)
    return r.status_code


if __name__ == "__main__":
    # for mgz in mgz_list:
    for i in page_range:
        url = host + mgz + "/recent?page=" + str(i)
        paper_links = get_paper_link(url)
        for link in paper_links:
            start_time = datetime.now()
            data = get_details(link)
            end_time = datetime.now()
            status_code = save2mongodb(data)
            if status_code == 200:
                success = 1
            else:
                success = 2
                print("已存在")
            print("链接：{}，第{}页，状态码：{}，用时：{}s".format(data["DOI"], i, success, end_time - start_time))
            random_sleep(1, 0.5)
        print("next page")
        random_sleep(30, 5)
