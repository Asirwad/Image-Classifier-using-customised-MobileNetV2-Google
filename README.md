# cat🐈‍⬛ vs. Dog🐕‍🦺 Image Classifier using Customized MobileNetV2 pre-trained model🔎
<table align="center" border=3>
  <tr>
    <td>
      <p align="center">
        <img src="https://cdn.dribbble.com/users/710395/screenshots/2504947/cat_dog_byb_dribble.gif" alt="alt" width="300">
      </p>
    </td>
  </tr>
</table>

In this project, we will build an image classifier to classify images of dogs and cats using a small dataset called `cats_vs_dogs` from the `tensorflow_datasets` module. The dataset contains (image, label) pairs, where images have different dimensions and three color channels.

## Data Preprocessing

Since the sizes of the images in the dataset are different, we will create a function to resize all images to the same size for consistency.

## Picking a Pretrained Model - MobileNet V2

For our image classification task, we will use the MobileNet V2 model, developed by Google. We will only use the convolutional base of the model and exclude the top (classification) layer. This base_model will output a shape of `(32, 5, 5, 1280)`, representing feature extraction from the original `(1, 160, 160, 3)` image. The 32 indicates that we have 32 layers of different filters/features.

## Model Architecture
<div>
  <details>
    <summary>
      🪄 <b>Customized model</b>
    </summary>
        <br />
      <p align="center">
        <table>
          <tr>
            <td><b><i>Customized model</i></b></td>
          </tr>
          <tr>
            <td>
              <p align="center">
                <img src="app/models/added_component.svg" alt="alt" width="200">
              </p>
            </td>
          </tr>
      </table>
    </p>
  </details>
</div>

<div>
  <details>
    <summary>
      🤖 <b>Base Model</b>
    </summary>
        <br />
      <p align="center">
        <table>
          <tr>
            <td><b><i>MobileNetV2</i></b></td>
          </tr>
          <tr>
            <td>
              <p align="center">
                <img src="app/models/base_model.svg" alt="alt" width="200">
              </p>
            </td>
          </tr>
      </table>
    </p>
  </details>
</div>

## Adding the Classifier

Instead of flattening the feature map of the base layer, we will use a global average pooling layer. This layer averages the entire 5x5 area of each 2D feature map and returns a single 1280-element vector per filter.

We will fine-tune the model on our dogs vs. cats dataset to achieve accurate classifications.



