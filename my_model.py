import math
import re
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

# a = "/home/nitish/Documents/Aishwarya_udacity/data/IMG/center_2021_04_21_12_34_05_080.jpg"
# times = '_2021_04_18'
# print(a.split('_2021')[0] + times + a.split('_21')[1])
# for line in reader:

    #     line.split('_2021')[0] + times + line.split('_21')[1]
    #     image_lines.append(line)
 
#################################################################################################
#################################################################################################

directory = './data/IMG/'
data_lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # i = 0
    for lines in reader:
        # i += 1
        # if i > 1000:
        #     break

        data_lines.append(lines)

# Acquiring the image from center, right and left cameras and steering angle

image_data = []
flipped_image = []
steering_angle = []

train_set, validation_Set = train_test_split(data_lines, test_size=0.3, train_size=None)

# collecting steering angles

for line in data_lines:

    steer = float(line[3])
    steering_angle.append(steer)
    steering_angle.append(steer+0.2)
    steering_angle.append(steer-0.2)

    # collecting images from left center and right cameras

    for columns in range(3):
        image_path = line[columns]
        filename = image_path.split('/')[-1]
        relative_path = directory + filename
        image = cv2.imread(relative_path)
        image_data.append(image)
        flipped_image.append(cv2.flip(image, 1))
        
print("Total Images: ", len(image_data)," | Total steering angles: ",len(steering_angle))

# Data Augmentation
augmented_image_data = image_data + flipped_image
flipped_steering_angle = []

for i in steering_angle:
    flipped  = -i
    flipped_steering_angle.append(flipped)
augmented_steering_angle = steering_angle + flipped_steering_angle
print("Augmented Size: ", len(augmented_steering_angle))

# Plotting the images captured
f, ax = plt.subplots(1,3, figsize=(36,5))
ax[0].imshow(image_data[0])
ax[0].set_title('Center Camera', fontsize=20)

ax[1].imshow(image_data[1])
ax[1].set_title('Left Camera', fontsize=20)

ax[2].imshow(image_data[3])
ax[2].set_title('Right Camera', fontsize=20)
plt.show()
print(image_data[0].shape)

# print(len(image_data))

# Data visualization
# no_bins = 31
# histogram, bins = np.histogram(augmented_steering_angle, no_bins)
# width = 0.5*(no_bins[1]-no_bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# mean_val = np.mean(histogram)

# keep = []
# for x in range(no_bins):
#     if histogram[x]<= mean_val:
#         keep.append(1)
#     else:
#         keep.append(mean_val/histogram(x))


nbin = 25
hist, bins = np.histogram(augmented_steering_angle, nbin)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.savefig("./examples/old_hist.png")
plt.show()
avg_value = np.mean(hist)
print("Mean of Measurement: ", avg_value)

# keep_prob = []
# for i in range(nbin):
#     if hist[i] <= avg_value:
#         keep_prob.append(1)
#     else:
#         keep_prob.append(avg_value/hist[i])

# remove_ind = []
# for i in range(len(augmented_steering_angle)):
#     for j in range(nbin):
#         if augmented_steering_angle[i] > bins[j] and augmented_steering_angle[i] <= bins[j+1]:
#             if random.random() < (1-keep_prob[j]):
#                 remove_ind.append(i)


# new_images = np.delete(augmented_image_data, remove_ind, 0)
# new_measurements = np.delete(augmented_steering_angle, remove_ind, 0)
    # new_images = aug_images
    # new_measurements = aug_measurements

# print("Saving pickle... ")
    
# with open('data.pkl','wb') as f:
#     pickle.dump([new_images, new_measurements], f)

# nbin = 25
# hist, bins = np.histogram(new_measurements, nbin)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.savefig("./examples/new_hist.png")
# plt.show()

# w = 0.5
# n = 30

# ax = plt.hist(augmented_steering_angle, bins = n)
# plt.show()

    
        
