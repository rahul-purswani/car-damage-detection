# Car Damage Detection on Edge
### Overview
The Car Damage Detection project focuses on developing a robust model to automatically detect and classify car damages, with the additional capability of deployment on an ESP32S board for real-time inference. The goal is to streamline the assessment of car damages, which can be particularly useful in scenarios such as accident evaluations or insurance claims. The model was succesfully deployed and was successful in detecting the damages in real-time as shown in demos below.

<h>For more details, refer to the [presentation](https://github.com/rahul-purswani/car-damage-detection/blob/main/Presentation.pdf).<h>

### Deployment Demo
![IMG_0182](https://github.com/rahul-purswani/car-damage-detection/assets/70603471/262da038-1ac9-45e3-b3d5-a8eea2a02a4d)
![IMG_0181](https://github.com/rahul-purswani/car-damage-detection/assets/70603471/2dd93b54-29b0-4dbf-ba65-94babf7c6c2a)
![IMG_0184](https://github.com/rahul-purswani/car-damage-detection/assets/70603471/0d7c61b9-b27e-41c5-8669-a37c76a7a85b)

### Good Detections
Below are some examples of successful detections made by the model, more examples can be found under good detections folder and inference.ipynb.
<img width="1119" alt="Screenshot 2024-03-29 at 10 42 55 PM" src="https://github.com/rahul-purswani/car-damage-detection/assets/70603471/108190ab-d70c-49d6-8773-d79280d5c514">

### Bad Detections
Below are some examples of unsuccessful detections made by the model, more examples can be found under bad detections folder and inference.ipynb.
<img width="1026" alt="Screenshot 2024-03-29 at 10 44 52 PM" src="https://github.com/rahul-purswani/car-damage-detection/assets/70603471/7d7d620c-fa5a-4890-9a63-e333f94c5fd7">

### Training Workflow
The training workflow involved several key steps. First, the necessary dependencies were installed to set up the TensorFlow Object Detection API environment. Pretrained model weights for MobileNetV2_ssd were then loaded to leverage existing knowledge. The training pipeline was configured, and the model was trained for 70,000 steps. Post-training quantization was applied to optimize the model, followed by converting the trained model to TensorFlow Lite (TFLite) format. Finally, the TFLite model was converted to a .cc file for deployment on the ESP32S board, enabling real-time car damage detection.
<h>For more details, refer to the [presentation](https://github.com/rahul-purswani/car-damage-detection/blob/main/Presentation.pdf).<h>

### References
M. Muktar, ‘CarDD-ReallyReal Dataset’, Roboflow Universe. Roboflow, Oct-2023.
```
@misc{ cardd-reallyreal_dataset,
    title = { CarDD-ReallyReal Dataset },
    type = { Open Source Dataset },
    author = { Moizuddin muktar },
    howpublished = { \url{ https://universe.roboflow.com/moizuddin-muktar-stt7g/cardd-reallyreal } },
    url = { https://universe.roboflow.com/moizuddin-muktar-stt7g/cardd-reallyreal },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { oct },
    note = { visited on 2023-11-29 },
}
```
X. Wang, W. Li and Z. Wu, "CarDD: A New Dataset for Vision-Based Car Damage Detection," in IEEE Transactions on Intelligent Transportation Systems, vol. 24, no. 7, pp. 7202-7214, July 2023, doi: 10.1109/TITS.2023.3258480.
```
@ARTICLE{10078726,
  author={Wang, Xinkuang and Li, Wenjing and Wu, Zhongcheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={CarDD: A New Dataset for Vision-Based Car Damage Detection}, 
  year={2023},
  volume={24},
  number={7},
  pages={7202-7214},
  doi={10.1109/TITS.2023.3258480}}
```
