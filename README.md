# ganless-hd
This codebase implements a method to train a neural network to produce high resolution (512x512) images WITHOUT utilizing a GAN. 

Input into the network are sparse, contour lines and (optionally) a low resolution (16x16) colormap. Check out the dataloader.py for details. 

The basic idea is to utilize a U-Net, modified perceptual loss (pearson instead of MAE), learned basis functions, and  "mean teacher" training in order to synthesize images of high quality without the normal troubles of training a GAN.

Video describing the method: https://www.youtube.com/watch?v=IdgEBYd5FfU

To run you need a 12GB GPU, pytorch .40, python 3.

python train.py

You will need to update the code in train.py to reflect the path(s) to your dataset:
```python
parser.add_argument('--dataroota', default=[
	'/path/to/your/dataset',
	], type=str)
```
By default it will utilize a network I trained on MS-COCO. If you want to start from scratch comment out (in train.py):
```python
loadedSD = torch.load('./saves/autoEncoder--3.983832822715064.pth')
for k in netGStateDict.keys():
	print(k)
	if k in loadedSD and netGStateDict[k].size() == loadedSD[k].size():
		netGStateDict[k] = loadedSD[k]
		print('... copied')
autoEncoder.load_state_dict(netGStateDict  )
```
Example image output, after training: (Column order-> INPUT, OUTPUT, TARGET)
https://github.com/lolz0r/ganless-hd/blob/master/images/output-1.png

### Techincal Details ###
## Learned basis functions ##
The initial basis functions are a product of SVD on a pretrained (resnet18, imagenet) neural network. These 'basis functions' are then further tuned, per layer, inside of ConvSeluSVD

Check out the code for ConvSeluSVD inside of model.py for how it is implemented.

## Modified perceptual loss ##
Pearson distance, instead of MSE/MAE, is used. Checkout function pearsonr inside of train.py

## Mean Teacher ##
A running, exponetial average of the last N (5) weights are used to calculate the 'next' set of weights. In train.py:

```python
if len(lastNWeights) > meanToStart:
		meanTeacher = np.array(lastNWeights)
		meanTeacher = np.average(meanTeacher, axis=0, weights=meanWeights).astype(np.float32)
		autoEncoder.inject_parameters(meanTeacher)
		del lastNWeights[0]
```



