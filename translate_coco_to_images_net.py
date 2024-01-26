def translate_coco_to_images_net(searched_label):
    label = ""
    if (searched_label in ["n02105855", "n02114548", "n02089973", "n02097209", "n02111277", "n02112018", "n02093859"]) :
        label = "dog"
    elif (searched_label in ["n02930766", "n03930630"]) :
        label = "car"
    elif (searched_label in ["n07742313"]) :
        label = "apple"
    return label