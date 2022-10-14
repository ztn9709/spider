import re
import time
import random
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from retry import retry

from selenium import webdriver
from selenium.webdriver.chrome.service import Service

mgz_list = ["prb", "prl", "pra", "prresearch", "rmp"]
page_range = range(107, 200)


chrome_path = "C:\\ProgramData\\Anaconda3\\chromedriver.exe"
options = webdriver.ChromeOptions()
options.page_load_strategy = "eager"
options.add_argument("headless")
options.add_argument("--ignore-ssl-errors")
options.add_argument("--ignore-certificate-errors")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument("--disable-blink-features=AutomationControlled")
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
    for result in soup.select("#search-result-list .title a"):
        link = "https://journals.aps.org" + result.attrs["href"]
        paper_links.append(link)
    driver.close()
    driver.quit()
    return paper_links


@retry()
def get_details(url):
    driver = webdriver.Chrome(service=Service(chrome_path), options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "lxml")
    driver.close()
    driver.quit()
    data = {
        "link": url,
        "title": "",
        "authors": [],
        "institutes": [],
        "DOI": "",
        "abstract": "",
        "areas": [],
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
    data["date"] = soup.find(attrs={"property": "article:published_time"}).attrs[
        "content"
    ]
    if soup.select(".physh-concepts"):
        for area in soup.select(".physh-concepts")[0].select(".physh-concept"):
            data["areas"].append(area.text)
    data["publication"] = data["DOI"].split("/")[-1].split(".")[0]
    return data


def is_topo(data):
    if "Topological phases of matter" in data["areas"]:
        return 1
    topo_areas = [
        "Symmetry protected topological states",
        "Topological insulators",
        "Topological order",
        "Topological phase transition",
        "Topological superconductors",
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
    total = 0
    success = 0
    for i in page_range:
        url = "https://journals.aps.org/" + "prresearch" + "/recent?page=" + str(i)
        paper_links = get_paper_link(url)
        for link in paper_links:
            start_time = datetime.now()
            data = get_details(link)
            end_time = datetime.now()
            total = total + 1
            if is_topo(data):
                status_code = save2mongodb(data)
                if status_code == 200:
                    success = 1
                else:
                    success = 2
                    print("已存在")
            print(data["DOI"], i, success, end_time - start_time)
            random_sleep(1, 0.5)
            success = 0
        print("next page")
        random_sleep(30, 5)