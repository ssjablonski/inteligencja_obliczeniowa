# importing the required library  
from imageai.Detection import ObjectDetection  
  
# instantiating the class  
recognizer = ObjectDetection()  
  
# defining the paths  
path_model = "./Models/yolo-tiny.h5"  
path_input = "./Input/test.jpg"  
path_output = "./Output/newimage.jpg"  
  
# using the setModelTypeAsTinyYOLOv3() function  
recognizer.setModelTypeAsTinyYOLOv3()  
# setting the path of the Model  
recognizer.setModelPath(path_model)  
# loading the model  
recognizer.loadModel()  
# calling the detectObjectsFromImage() function  
recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  
  
# iterating through the items found in the image  
for eachItem in recognition:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  