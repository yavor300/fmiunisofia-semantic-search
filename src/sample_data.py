"""
Sample data generator for testing the search engine.
"""

import pandas as pd
import os


def generate_sample_data(output_path: str = "data/sample_products.csv"):
    """
    Generate sample product data for testing.
    
    Args:
        output_path: Path to save the CSV file
    """
    
    sample_products = [
        # Budget bikes
        {
            "title": "Basic City Bike 26 inch Steel Frame",
            "brand": "Generic",
            "description": "Simple and reliable city bike for daily commuting. Steel frame, single speed.",
            "final_price": 199.99,
            "category": "Bike",
            "weight": 15.5
        },
        {
            "title": "Urban Commuter Bicycle Black",
            "brand": "CityRide",
            "description": "Affordable urban bike with comfortable seat and basket. Perfect for short trips.",
            "final_price": 249.99,
            "category": "Bike",
            "weight": 14.2
        },
        
        # Mid-range bikes
        {
            "title": "Drag Grand Canyon Mountain Bike",
            "brand": "Drag",
            "description": "27.5 inch MTB with Shimano components. Great for trails and off-road adventures.",
            "final_price": 599.99,
            "category": "MTB",
            "weight": 13.8
        },
        {
            "title": "Trek Marlin 7 Mountain Bike",
            "brand": "Trek",
            "description": "Versatile mountain bike with 29 inch wheels. Aluminum frame, excellent for beginners.",
            "final_price": 899.99,
            "category": "MTB",
            "weight": 13.2
        },
        
        # Premium bikes
        {
            "title": "Specialized Rockhopper Expert 29",
            "brand": "Specialized",
            "description": "High-performance trail bike with carbon fork and hydraulic disc brakes.",
            "final_price": 1499.99,
            "category": "MTB",
            "weight": 11.9
        },
        {
            "title": "Giant Defy Advanced Pro Road Bike",
            "brand": "Giant",
            "description": "Professional carbon road bike for serious cyclists. Ultra-lightweight design.",
            "final_price": 3299.99,
            "category": "Road",
            "weight": 8.5
        },
        
        # Running shoes - budget
        {
            "title": "Basic Running Shoes Men Black",
            "brand": "SportMax",
            "description": "Comfortable running shoes for casual joggers. Breathable mesh upper.",
            "final_price": 39.99,
            "category": "Shoes",
            "weight": 0.6
        },
        {
            "title": "Budget Athletic Sneakers",
            "brand": "FitPro",
            "description": "Affordable athletic shoes for gym and light running. Good cushioning.",
            "final_price": 49.99,
            "category": "Shoes",
            "weight": 0.65
        },
        
        # Running shoes - mid-range
        {
            "title": "Nike Revolution 6 Running Shoes",
            "brand": "Nike",
            "description": "Reliable running shoes with excellent cushioning. Suitable for daily training.",
            "final_price": 89.99,
            "category": "Shoes",
            "weight": 0.55
        },
        {
            "title": "Adidas Ultraboost 22 Running",
            "brand": "Adidas",
            "description": "Premium running shoes with Boost technology. Responsive and comfortable.",
            "final_price": 189.99,
            "category": "Shoes",
            "weight": 0.62
        },
        
        # Electronics - laptops
        {
            "title": "Budget Laptop 14 inch Intel Celeron",
            "brand": "TechBasic",
            "description": "Entry-level laptop for basic tasks. 4GB RAM, 128GB SSD. Perfect for students.",
            "final_price": 299.99,
            "category": "Laptop",
            "weight": 1.5
        },
        {
            "title": "HP Pavilion 15 Laptop Intel i5",
            "brand": "HP",
            "description": "Mid-range laptop with 8GB RAM, 512GB SSD. Great for work and entertainment.",
            "final_price": 699.99,
            "category": "Laptop",
            "weight": 1.75
        },
        {
            "title": "Dell XPS 13 Premium Ultrabook",
            "brand": "Dell",
            "description": "High-end ultraportable laptop. Intel i7, 16GB RAM, 1TB SSD. Beautiful display.",
            "final_price": 1599.99,
            "category": "Laptop",
            "weight": 1.2
        },
        {
            "title": "Apple MacBook Pro 14 M2 Pro",
            "brand": "Apple",
            "description": "Professional laptop for creators. M2 Pro chip, 16GB RAM, stunning Retina display.",
            "final_price": 2499.99,
            "category": "Laptop",
            "weight": 1.6
        },
        
        # Headphones
        {
            "title": "Budget Wireless Earbuds",
            "brand": "SoundMax",
            "description": "Affordable Bluetooth earbuds with decent sound quality. 5 hour battery.",
            "final_price": 29.99,
            "category": "Headphones",
            "weight": 0.05
        },
        {
            "title": "Sony WH-1000XM5 Noise Cancelling",
            "brand": "Sony",
            "description": "Premium over-ear headphones with industry-leading noise cancellation.",
            "final_price": 399.99,
            "category": "Headphones",
            "weight": 0.25
        },
        
        # Bikes with specific features
        {
            "title": "Family Bike with Child Seat",
            "brand": "FamilyRide",
            "description": "Sturdy city bike with integrated child seat. Safety-tested and reliable.",
            "final_price": 449.99,
            "category": "Bike",
            "weight": 18.5
        },
        {
            "title": "Cross Country Racing Bike Ultra Light",
            "brand": "Cross",
            "description": "Competition-ready XC bike. Carbon frame, top-tier components.",
            "final_price": 4999.99,
            "category": "MTB",
            "weight": 9.2
        },
        {
            "title": "Electric Mountain Bike E-Bike",
            "brand": "Trek",
            "description": "Electric MTB with powerful motor. Perfect for hills and long distances.",
            "final_price": 3499.99,
            "category": "E-Bike",
            "weight": 22.5
        },
        {
            "title": "Folding Bike Compact Portable",
            "brand": "Brompton",
            "description": "Ultra-portable folding bike. Great for commuters with limited storage.",
            "final_price": 1299.99,
            "category": "Bike",
            "weight": 11.5
        },
    ]
    
    df = pd.DataFrame(sample_products)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Sample data generated: {output_path}")
    print(f"Total products: {len(df)}")
    
    return output_path


if __name__ == "__main__":
    generate_sample_data()
