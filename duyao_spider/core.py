#!/usr/bin/env python3
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from tqdm import tqdm

from ocr import ocr, utils

UPDATE_IMAGES = False
IMG_INFO = Path("downloaded_imgs.json")
IMAGES_STORE = Path("duyaoSS_images")
CSV_STORE = Path("duyaoSS_tables")

if __name__ == "__main__":
    # Update images
    if UPDATE_IMAGES:
        if IMG_INFO.is_file():
            IMG_INFO.unlink()
        spider = CrawlerProcess(get_project_settings())
        spider.crawl("DuyaoSS")
        spider.start()

    # Go through results
    CSV_STORE.mkdir(exist_ok=True)

    img_info = utils.load_crawl_results(IMG_INFO)
    for item in tqdm(img_info):
        provider, imgs = item["provider"], item["images"]
        provider = utils.remove_provider_numbering(provider)

        print(f"Parsing content for provider: {provider}")
        for i, f in enumerate(imgs):
            img = ocr.read_img(str(IMAGES_STORE / f["path"]))
            rows, footer = ocr.img_to_csv(img)
            res = utils.csv_to_df(
                rows, provider, save=CSV_STORE / f"{provider}_{i}.csv"
            )
