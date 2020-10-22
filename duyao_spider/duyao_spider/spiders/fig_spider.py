import scrapy
from duyao_spider.items import DuyaoSpiderItem


class DuyaossSpider(scrapy.Spider):
    name = "DuyaoSS"
    start_urls = ["https://www.duyaoss.com/archives/1031/"]

    def parse(self, response):
        # first find numbered titles (1. xxx, 2.xxx etc.)
        title_xpath = "h2[re:match(text(), '\d+\.')]"
        providers = response.xpath(f"//{title_xpath}")

        # for each title, get all images before the next title
        # it's done this way because the figures are not children nodes but siblings
        for i, h2 in enumerate(providers, start=1):
            figs = h2.xpath(
                f"./following-sibling::p[count(preceding-sibling::{title_xpath})={i}]/figure"
            )
            yield DuyaoSpiderItem(
                provider=h2.xpath("text()").get(),
                image_urls=figs.xpath(".//img/@data-src").getall(),
            )
