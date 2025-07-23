from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
from PIL import Image
from textblob import TextBlob

def detect_and_caption(image_path):
    # Open image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    CONFIDENCE_THRESHOLD = 0.7

    # Load models
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    Blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    YOLO_model = YOLO("ml/yolo11n.pt")
    
    # Predicting captions with Blip2
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = Blip_model.generate(**inputs)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Predicting foods in image with YOLO11
    results = YOLO_model(image)
    for result in results:
        boxes = result.boxes
        names = [
            result.names[cls.item()]
            for cls, conf in zip(boxes.cls.int(), boxes.conf)
            if conf.item() >= CONFIDENCE_THRESHOLD
        ]

    # Adds all nouns from the caption to the list of ingredients
    # which will be filtered later
    ingredients = set(names)
    ingredients.update(TextBlob(generated_text).noun_phrases)
    ingredients = list(ingredients)
    
    # Prints results to terminal and returns a tuple containing a list
    # of the objects detected, and the caption for the image
    print(names)    
    print(generated_text)
    print(ingredients)
    return ingredients

def extract_ingredients(ingredients):
    whitelist = {
        # Fruits
        "apple", "banana", "orange", "grape", "strawberry", "blueberry", "kiwi", "pineapple",
        "lemon", "lime", "mango", "pear", "peach", "cherry", "watermelon", "coconut", "avocado",

        # Vegetables
        "carrot", "potato", "tomato", "onion", "garlic", "lettuce", "spinach", "cucumber",
        "zucchini", "broccoli", "cauliflower", "pepper", "bell pepper", "green bean", "corn",
        "cabbage", "celery", "eggplant", "mushroom", "radish", "beet", "asparagus",

        # Grains & Legumes
        "rice", "bread", "pasta", "noodle", "tortilla", "oats", "quinoa", "barley",
        "lentil", "chickpea", "bean", "black bean", "pinto bean", "soybean", "wheat", "cornmeal",

        # Proteins
        "chicken", "beef", "pork", "turkey", "bacon", "ham", "sausage", "salmon", "tuna",
        "shrimp", "crab", "lobster", "egg", "steak", "meatball", "duck", "tofu",

        # Dairy
        "milk", "cheese", "butter", "yogurt", "cream", "ice cream", "parmesan", "mozzarella",
        "sour cream", "cream cheese",

        # Spices & Herbs
        "salt", "pepper", "cinnamon", "nutmeg", "ginger", "turmeric", "oregano", "basil",
        "thyme", "rosemary", "parsley", "cilantro", "chili", "paprika", "curry powder",

        # Condiments & Sauces
        "ketchup", "mustard", "mayonnaise", "soy sauce", "vinegar", "hot sauce", "bbq sauce",
        "honey", "maple syrup", "salsa", "pesto", "tomato sauce", "ranch", "vinaigrette",

        # Oils & Fats
        "olive oil", "vegetable oil", "canola oil", "coconut oil", "sesame oil", "lard",

        # Baking
        "flour", "sugar", "brown sugar", "baking powder", "baking soda", "yeast", "vanilla",

        # Misc
        "pickle", "jam", "jelly", "chocolate", "peanut butter", "almond", "cashew", "walnut",
        "hazelnut", "raisin", "granola", "cereal", "crouton", "breadcrumbs"
    }   

    for ingredient in ingredients:
        if ingredient.lower() not in whitelist:
            ingredients.remove(ingredient)
            
    print(ingredients)        
    return ingredients

# Used to make sure the function works, will remove later!
detect_and_caption("/home/wlazlo/Portfolio-Project/Food4Recipes/sample-images/apples.jpg")
extract_ingredients(detect_and_caption("/home/wlazlo/Portfolio-Project/Food4Recipes/sample-images/fruits.jpg"))