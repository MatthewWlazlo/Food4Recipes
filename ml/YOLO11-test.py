from ultralytics import YOLO

# Load model
model = YOLO("ml/yolo11n.pt")
        
def fruit_finder(path):
    counter = 1
    results = model(path)
    
    # Prints out the class of each object detected, and the number of objects detected
    for result in results:
        names = model.names 
        boxes = result.boxes
        # Iterates through each of the class IDs in boxes for printing
        for cls_id in boxes.cls.tolist():
            print(f"{counter}. {names[int(cls_id)]}")
            counter += 1
            
            
fruit_finder("/home/wlazlo/Portfolio-Project/Food4Recipes/sample-images/apples.jpg")

'''
NOTES:
The first image counts six apples because it identifies the reflection
of the apples as apples. Important to keep in mind. Should warn users
taking pictures on reflective surfaces.

YOLO11 does not identify grapes and kiwi based on this little test 
(it's trained on the COCO dataset which only has 10 food classes), so
that's  something that will need to be addressed as well. Will definitely
need a feature that allows users to explicitly label an ingredient, and
delete mislabeled ingredients.
'''