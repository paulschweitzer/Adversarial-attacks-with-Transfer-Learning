import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, EfficientNetB0, MobileNetV3Small, MobileNetV2
from tensorflow.keras.applications import resnet50, vgg16, inception_v3, mobilenet_v2, mobilenet_v3

all_models = {
    "ResNet50": (ResNet50(weights='imagenet'), resnet50.preprocess_input),
    "VGG16": (VGG16(weights='imagenet'), vgg16.preprocess_input),
    "MobileNetV2": (MobileNetV2(weights='imagenet'), mobilenet_v2.preprocess_input),
    # "MobileNetV3Small": (MobileNetV3Small(weights='imagenet'), mobilenet_v3.preprocess_input),
    "InceptionV3": (InceptionV3(weights='imagenet'), inception_v3.preprocess_input),
    }


def preprocess(image, preprocess_input, size=(224,224)):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, size)
  image = preprocess_input(image)
  image = image[None, ...]
  return image


def deprocess_blue(image):
    # Assume shape (H, W, 3) or (1, H, W, 3)
    if image.shape.rank == 4:
        image = image[0]
    image = image[..., ::-1]  # BGR → RGB
    image = image + [103.939, 116.779, 123.68]  # Unsubtract means
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return image


def save_image(tensor, target_folder, filename):

    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    # Encode as JPEG
    encoded = tf.io.encode_jpeg(tensor)

    # Ensure the directory exists
    os.makedirs(target_folder, exist_ok=True)

    # Save to file
    path = os.path.join(target_folder, filename)
    tf.io.write_file(path, encoded)
    # print(f"Image saved to: {path}")

### Untargeted perturbation
def create_adversarial_pattern(input_image, input_label, loss_object, model):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


def generate_untargeted_adversarial_images_multiple_models(images, source_folder, target_folder, label, eps, models):

    i, mx = 0, len(images)
    label = tf.one_hot(label, 1000)
    label = tf.reshape(label, (1, 1000))
    adversarial_tensors = []

    for img_name in images:
        # print(f"{i + 1}/{mx}: {img_name}")
        # Get image as tensor
        image_raw = tf.io.read_file(source_folder + img_name)
        image = tf.image.decode_image(image_raw)

        # Iterate through each model
        for j, m in enumerate(models.keys()):
            model = models[m][0]
            preprocessor = models[m][1]
            
            # Preprocess image depending on the model
            if m == "InceptionV3":
                image_pr = preprocess(image, preprocessor, (299, 299))
            elif m == "MobileNetV3" or m == "MobileNetV3Small":
                image_pr = preprocess(image, lambda x: x)
            else:
                image_pr = preprocess(image, preprocessor)

            # Create untargeted perturbation
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            pertubation = create_adversarial_pattern(image_pr, label, loss_object, model)

            # Apply perturbation and clip values
            if m == "VGG16" or m == "ResNet50":
                # [-128, 128]
                adv = image_pr + 100 * eps * pertubation
                adv = tf.clip_by_value(adv, -128, 128)
                final = (deprocess_blue(adv) - 0.5) * 2

            elif m == "MobileNetV2" or m == "InceptionV3":
                # [-1, 1]
                adv = image_pr + eps * pertubation
                final = tf.clip_by_value(adv, -1, 1)

            elif m == "MobileNetV3" or m == "MobileNetV3Small":
                adv = image_pr + 255 * eps * pertubation
                final = tf.clip_by_value(adv, 0, 255)

            # Convert from [-1, 1] to [0, 255]
            tensor = (final + 1.0) / 2.0
            tensor = tf.clip_by_value(tensor, 0.0, 1.0)
            tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8)

            if len(tensor.shape) == 4:
                tensor = tensor[0]

            image = tensor

        # Save image
        save_image(tensor, target_folder, img_name)
        adversarial_tensors.append(tensor)

        i += 1
        
    return adversarial_tensors

### Targeted perturbation
# For clipping the input vector
def clip_delta(t, eps):
    return tf.clip_by_value(t, -eps, eps)


# Generates targeted adversarial images
def generate_adv_pertubation(model_name, model, preprocessor, optimizer, loss, img, delta, label, target_label, eps, max_iterations=500):
    # delta = tf.Variable(tf.zeros_like(img), trainable=True)
    
    for step in range(max_iterations):
        with tf.GradientTape() as tape:
            tape.watch(delta)
            
            if model_name == "InceptionV3":
                adv = preprocess(img + delta, preprocessor, (299, 299))
            else:
                adv = preprocess(img + delta, preprocessor)

            preds = model(adv,training=False)
            # print(preds)
            originalLoss = -loss(tf.convert_to_tensor([label]), preds)
            targetLoss = loss(tf.convert_to_tensor([target_label]), preds)
            totalLoss = originalLoss + targetLoss

            if step % 50 == 0:
                print("step: {}, loss: {}...".format(step, totalLoss.numpy()), "Range:", tf.reduce_min(adv).numpy(), "to", tf.reduce_max(adv).numpy())

            gradients = tape.gradient(totalLoss, delta)
            optimizer.apply_gradients([(gradients, delta)])
            delta.assign_add(clip_delta(delta, eps))
    # tensor_range(delta)
    
    return delta


def generate_adversarial_images(images, models, source_folder, target_folder, label, target_label, eps, iterations):
    i, mx = 0, len(images)
    adversarial_imgs = []
    for img_name in images:
        print(f"{i}/{mx}: {img_name}")
        
        image_raw = tf.io.read_file(source_folder + img_name)
        image = tf.image.decode_image(image_raw)
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        adam = tf.keras.optimizers.Adam(1e-2)
        
        baseImage = tf.constant(image / 1, dtype=tf.float32)
        delta = tf.Variable(tf.zeros_like(baseImage), trainable=True)

        for m in models.keys():
            
            model = models[m][0]
            preprocessor = models[m][1]

            target_pertubation = generate_adv_pertubation(m, model, preprocessor, adam, loss, baseImage, delta, label, target_label, eps, iterations)
            adv = (baseImage + target_pertubation)

            # baseImage = tf.clip_by_value(adv, val1, val2)
            baseImage = adv

        final = adv
        if m == "InceptionV3":
            final = preprocess(final, preprocessor, (299, 299))
        else:
            final = preprocess(final, preprocessor)
        # _, cl, conf = get_imagenet_label(model.predict(final))
        # display_images(adv, descriptions[i])
        # print(f"Classified as {cl} with {conf}% confidence.")
        
        if m == "VGG16" or m == "ResNet50":
            final = (deprocess_blue(final) - 0.5) * 2

        # Convert from [-1, 1] to [0, 255]
        tensor = (final + 1.0) / 2.0
        tensor = tf.clip_by_value(tensor, 0.0, 1.0)
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8)
        save_image(tensor, target_folder, img_name)
        adversarial_imgs.append(tensor)
        # plt.imshow(tensor)
        # plt.title(f"After perturbation with model {m}")
        # plt.show()
        i += 1
    return adversarial_imgs

if __name__ == "__main__":
    full_path = "./banana_images/data/train/banana/"
    target_path = "./adversarial_images/targeted2"
    image_files = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    sub_images = image_files[:100]

    # model_name = "MobileNetV2"
    # model = {model_name: all_models[model_name]}

    eps = 0.03
    label = tf.Variable(954)
    target_label = tf.Variable(953)
    iterations = 150

    advs_targeted = generate_adversarial_images(sub_images, all_models, full_path, target_path, label, target_label, eps, iterations)