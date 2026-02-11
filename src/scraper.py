import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class Product:
    title: str
    url: str
    price: Optional[str]
    currency: Optional[str]
    vendor: Optional[str]
    sku: Optional[str]
    image: Optional[str]
    description: Optional[str]
    long_description: Optional[str]


def fetch_html(session: requests.Session, url: str, timeout: int = 20) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_collection_page(html: str, base_url: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "html.parser")
    product_urls = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if "/products/" not in href:
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        product_urls.add(parsed._replace(fragment="").geturl())

    return sorted(product_urls)


def parse_product_jsonld(soup: BeautifulSoup) -> dict:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "Product":
                    return item
        if isinstance(data, dict) and data.get("@type") == "Product":
            return data
    return {}


def parse_product_page(html: str, url: str) -> Product:
    soup = BeautifulSoup(html, "html.parser")
    jsonld = parse_product_jsonld(soup)

    title = (
        jsonld.get("name")
        or (soup.find("h1").get_text(strip=True) if soup.find("h1") else "")
    )

    price = None
    currency = None
    offers = jsonld.get("offers")
    if isinstance(offers, dict):
        price = offers.get("price")
        currency = offers.get("priceCurrency")
    elif isinstance(offers, list) and offers:
        price = offers[0].get("price")
        currency = offers[0].get("priceCurrency")

    vendor = None
    brand = jsonld.get("brand")
    if isinstance(brand, dict):
        vendor = brand.get("name")
    elif isinstance(brand, str):
        vendor = brand

    sku = jsonld.get("sku")

    image = None
    images = jsonld.get("image")
    if isinstance(images, list) and images:
        image = images[0]
    elif isinstance(images, str):
        image = images

    description = jsonld.get("description")
    if not description:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

    long_description = None
    long_desc_container = soup.select_one(".metafield-rich_text_field")
    if not long_desc_container:
        long_desc_container = soup.find(attrs={"role": "tabpanel"})
    if long_desc_container:
        long_description = long_desc_container.get_text(" ", strip=True)
        if not long_description:
            long_description = None

    return Product(
        title=title,
        url=url,
        price=str(price) if price is not None else None,
        currency=currency,
        vendor=vendor,
        sku=sku,
        image=image,
        description=description,
        long_description=long_description,
    )


def crawl_collection(
    collection_url: str,
    max_pages: int,
    delay: float,
    limit: Optional[int],
) -> list[Product]:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
            )
        }
    )

    products: list[Product] = []
    page = 1
    seen_urls = set()

    while page <= max_pages:
        page_url = f"{collection_url}?page={page}"
        html = fetch_html(session, page_url)
        product_urls = parse_collection_page(html, collection_url)

        if not product_urls:
            break

        for product_url in product_urls:
            if product_url in seen_urls:
                continue
            seen_urls.add(product_url)

            product_html = fetch_html(session, product_url)
            product = parse_product_page(product_html, product_url)
            products.append(product)

            if limit and len(products) >= limit:
                return products

            time.sleep(delay)

        page += 1
        time.sleep(delay)

    return products


def write_csv(products: list[Product], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "url",
                "price",
                "currency",
                "vendor",
                "sku",
                "image",
                "description",
                "long_description",
            ],
        )
        writer.writeheader()
        for product in products:
            writer.writerow(asdict(product))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape products from collection pages."
    )
    parser.add_argument(
        "--collection-url",
        default="https://dragzone.bg/collections/komponenti",
        help="Collection URL to crawl.",
    )
    parser.add_argument(
        "--output",
        default="scraped_products.csv",
        help="CSV file to write results to.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum number of collection pages to scan.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay (seconds) between requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of products.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = crawl_collection(
        collection_url=args.collection_url,
        max_pages=args.max_pages,
        delay=args.delay,
        limit=args.limit,
    )
    write_csv(products, args.output)
    print(f"Saved {len(products)} products to {args.output}")


if __name__ == "__main__":
    main()
