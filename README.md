# FIND PHONE

- The problem requires the network to predict the center location of a phone in an image within an accuracy of 0.05 normalized distance. 
- Due to the simplicity of the images provided, there was no need to use complicated networks. According to me a network in itself is not essential, traditional image recognition techniques are sufficient for this specific task. 

- Having realized that simple techniques are sufficient, the features that can be used to extract the phones are very low level (edges and shapes) I choose to use a custom CNN. 
- The network contains 2 Convolutional layers and 4 linear layers. The reason only 2 convolutional layers are used is to strick a balance between feature detection but at the same time to prevent spatial information from being lost. 
- The input to network is an image of size 128x128, and the output is a real valued vector of size 1x2.

___
___
## Usage
___
### Train and store the network
- If the find_phone folder is in the same directory
    ```bash
    python train_find_phone.py
    ```
- If the path needs to be supplied
    ```bash
    python train_find_phone.py -f ~/find_phone/
    ```
___
### Validation script
- Ensure that the script is being run from the root of the project, this to ensure that the model can be loaded. It prints to terminal the normalized center coordinate.
    ```bash
    python find_phone.py -f ~/find_phone/51.jpg
    ```
- It can not do batch image validation, as the feature was not requested. 

___
___
## Future updates
For better accuracy, networks such as YOLOv5 or SSD can be used. These two networks were not utilized for this problem as the accuracy was sufficiently high for the given dataset. **The smallest network that gets the job done is the best network**