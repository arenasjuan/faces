import os
import random
import pygame
from pygame.locals import *
import cv2
import numpy as np
from imutils import face_utils
import dlib
import config
import shutil

detector = dlib.get_frontal_face_detector()

def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst

def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    pointDict = {(p[0], p[1]):i for i,p in enumerate(points)}
    for p in points:
        # Check if point is inside rectangle
        if rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]:
            subdiv.insert((float(p[0]), float(p[1])))
    triangleList = subdiv.getTriangleList()
    delaunayTri = []
    for t in triangleList:
        pt = []
        pt.append(pointDict.get((t[0], t[1]), -1))
        pt.append(pointDict.get((t[2], t[3]), -1))
        pt.append(pointDict.get((t[4], t[5]), -1))
        if -1 not in pt:
            delaunayTri.append(pt)
    return delaunayTri


def normalize_image(im, points):
    # Define desired output coordinates
    output_coords = np.float32([[180,200], [420,200]])
    # Extract input coordinates from original image
    input_coords = np.float32([points[36], points[45]])
    # Calculate similarity transformation
    M = cv2.estimateAffinePartial2D(input_coords, output_coords)[0]
    # Apply transformation to image
    normalized_im = cv2.warpAffine(im, M, (600,600))
    # Apply transformation to points
    normalized_points = np.reshape(cv2.transform(np.reshape(points, [1, -1, 2]), M), [points.shape[0], 2])
    return normalized_im, normalized_points



def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])

    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    imgRect = alpha * warpImage1 + ((1.0 - alpha) * warpImage2)
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

def add_boundaries_and_midway_points(shape, img):
    (h, w) = img.shape[:2]
    
    # Add corners of the image
    corners = [(0, 0), (0, w), (h, 0), (h, w)]

    # Add midway points
    midways = [(0, w//2), (h, w//2), (h//2, 0), (h//2, w)]
    
    # Adding these points to shape
    for point in corners + midways:
        shape = np.vstack([shape, point])

    return shape



def morph_faces(filename1, filename2, alpha) :
    # Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\joker\coding\faces\shape_predictor_68_face_landmarks.dat")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    dets1 = detector(gray1, 1)
    dets2 = detector(gray2, 1)

    shape1 = face_utils.shape_to_np(predictor(gray1, dets1[0]))
    shape2 = face_utils.shape_to_np(predictor(gray2, dets2[0]))

    # Normalize images and points
    img1, shape1 = normalize_image(img1, shape1)
    img2, shape2 = normalize_image(img2, shape2)

    # Convert to float32
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Perform alpha blending on the entire image first
    imgMorph = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)

    # Add boundary and midway points
    shape1 = add_boundaries_and_midway_points(shape1, img1)
    shape2 = add_boundaries_and_midway_points(shape2, img2)

    # Convert to integer and ensure all points in shape are within the bounding rectangle
    shape = np.rint((np.array(shape1) * (1.0 - alpha)) + (np.array(shape2) * alpha)).astype(int)
    shape = np.maximum(np.minimum(shape, [img1.shape[1]-1, img1.shape[0]-1]), [0, 0])

    rect = (0, 0, img1.shape[1], img1.shape[0])

    dt = calculateDelaunayTriangles(rect, shape)


    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []

        for j in range(0, 3):
            t1.append(shape1[dt[i][j]])
            t2.append(shape2[dt[i][j]])
            t.append(shape[dt[i][j]])

        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    return imgMorph


def draw_centered_text(screen, text, percent_of_screen_vertical_position=50, percent_of_screen_height=5):
    screen_width, screen_height = screen.get_size()
    font_size = int(screen_height * percent_of_screen_height / 100)
    font = pygame.font.Font(None, font_size)

    # Split the text into paragraphs based on newline characters
    paragraphs = text.split('\n')

    lines = []
    for paragraph in paragraphs:
        # Split each paragraph into lines that can fit within the screen width
        words = paragraph.split(' ')
        current_line = ''
        for word in words:
            test_line = current_line + ' ' + word
            text_surface = font.render(test_line.strip(), True, (255, 255, 255))
            if text_surface.get_width() > screen_width:  # If the line is too wide
                # Start a new line with the current word
                lines.append(current_line)
                current_line = word
            else:
                # Otherwise, add the word to the current line
                current_line = test_line
        # Append the last line of the paragraph
        lines.append(current_line)

    # Draw each line in the center of the screen
    vertical_position = screen_height * percent_of_screen_vertical_position / 100
    for i, line in enumerate(lines):
        text_surface = font.render(line.strip(), True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen_width / 2, vertical_position + i * font_size))
        screen.blit(text_surface, text_rect)




def choose_gender(screen):
    """Prompts the user to choose a gender preference."""
    running = True
    gender = None
    while running and gender is None:
        screen.fill((0,0,0)) # Clear the screen
        draw_centered_text(screen, "Press Left for Males, Right for Females, Up for Both")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    gender = "Male"
                elif event.key == pygame.K_RIGHT:
                    gender = "Female"
                elif event.key == pygame.K_UP:
                    gender = "Both"
        pygame.display.flip()  # Update the display

    return gender


def choose_directory(gender):
    """Returns a list of directories corresponding to the chosen gender(s)."""
    if gender == "Male":
        return [config.male]
    elif gender == "Female":
        return [config.female]
    else:
        return [config.male, config.female]

def load_images(screen, image_paths):
    image_objs = []
    max_height = screen.get_height() // 2  # Max height for image, to fit within screen
    max_width = screen.get_width() // 2  # Max width for image, to fit within screen
    for path in image_paths:
        img = pygame.image.load(path)
        width, height = img.get_size()

        # Calculate aspect ratio-preserving dimensions within screen constraints
        if width > max_width:
            scale_factor = max_width / width
            width = max_width
            height = int(height * scale_factor)
        if height > max_height:
            scale_factor = max_height / height
            height = max_height
            width = int(width * scale_factor)
        
        img = pygame.transform.scale(img, (width, height))  # Rescale image
        image_objs.append(img)
    return image_objs

def draw_images(screen, images):
    screen_width, screen_height = screen.get_size()
    padding = screen_width * 0.1

    left_image, right_image = images
    left_rect = left_image.get_rect(center=(screen_width / 4, screen_height / 2))
    right_rect = right_image.get_rect(center=(3 * screen_width / 4, screen_height / 2))

    # Ensure that the images do not overlap by checking if the right edge of the left image is greater than the left edge of the right image.
    while left_rect.right + padding > right_rect.left:
        # If they do overlap, then we reduce the size of the images until they do not.
        for img in images:
            img_width, img_height = img.get_size()
            img = pygame.transform.scale(img, (img_width - 10, img_height - 10))
        left_rect = left_image.get_rect(center=(screen_width / 4, screen_height / 2))
        right_rect = right_image.get_rect(center=(3 * screen_width / 4, screen_height / 2))

    screen.blit(left_image, left_rect)
    screen.blit(right_image, right_rect)


def compile_file_names(directory_path):
    file_names = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))]
    return file_names


def main():
    pygame.init()
    pygame.display.set_caption('Faces')
    screen = pygame.display.set_mode((1200, 600), pygame.RESIZABLE)
    running = True

    draw_centered_text(screen, "Welcome to Faces\n\nYou will be shown a series of faces in pairs; for each pair, select the face you're more attracted to. The faces you choose will be combined to show you the average face you're attracted to. Press any key to continue.", 30)
    pygame.display.flip()
    waiting_for_key = True
    while waiting_for_key:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                waiting_for_key = False

    screen.fill((0, 0, 0))

    chosen_gender = choose_gender(screen)
    avg_image_name = "AvgAttraction.png"
    if os.path.exists(avg_image_name):
        os.remove(avg_image_name)
    fullscreen = False

    male_faces = compile_file_names(config.male)
    female_faces = compile_file_names(config.female)
    all_faces = male_faces + female_faces
    chosen_images = set()

    images = None
    last_selected_img_path = None
    just_returned = False
    while running:
        just_returned = False
        selected_img_path = None

        if chosen_gender == "Both":
            if images is None or len(chosen_images) == len(all_faces):
                images = random.sample(all_faces, 2) if len(all_faces) >= 2 else all_faces
        elif chosen_gender == "Female":
            if images is None or len(chosen_images) == len(female_faces):
                images = random.sample(female_faces, 2)
        else:
            if images is None or len(chosen_images) == len(male_faces):
                images = random.sample(male_faces, 2)

        img_objs = load_images(screen, images)
        screen.fill((0, 0, 0))
        draw_images(screen, img_objs)
        draw_centered_text(screen, "Which face are you more attracted to? Use left or right arrow keys to choose", 10)
        pygame.display.flip()

        running_selection = True
        while running_selection and running:
            if just_returned:
                just_returned = False
                running_selection = False
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    img_objs = load_images(screen, images)  # Reload images after resizing
                    screen.fill((0, 0, 0))
                    draw_images(screen, img_objs)  # Redraw images after resizing
                    draw_centered_text(screen, "Which face are you more attracted to? Use left or right arrow keys to choose", 10)
                    pygame.display.flip()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:  # F11 key to toggle fullscreen
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                        img_objs = load_images(screen, images)  # Reload images after resizing
                        screen.fill((0, 0, 0))
                        draw_images(screen, img_objs)  # Redraw images after resizing
                        draw_centered_text(screen, "Which face are you more attracted to? Use left or right arrow keys to choose", 10)
                        pygame.display.flip()
                    elif event.key == pygame.K_q:
                        previous_images = images.copy()
                        last_selected = last_selected_img_path 
                        showing_average = True
                        while showing_average:
                            screen.fill((0, 0, 0))
                            avg_image = pygame.image.load("AvgAttraction.png").convert_alpha()
                            screen.blit(avg_image, ((screen.get_width() - avg_image.get_width()) / 2, (screen.get_height() - avg_image.get_height()) / 2))
                            draw_centered_text(screen, "Here is the average apple of your eye\n\nPress 'Q' again to quit Faces, or 'C' to continue Faces", 10)
                            pygame.display.flip()

                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                elif event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_q:
                                        pygame.quit()
                                        return
                                    elif event.key == pygame.K_c:
                                        images = previous_images
                                        showing_average = False
                                        selected_img_path = last_selected_img_path
                                        running_selection = False
                                        just_returned = True
                                        break
                    elif event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        selected_img_index = 0 if event.key == pygame.K_LEFT else 1
                        selected_img_path = images[selected_img_index]
                        last_selected_img_path = selected_img_path
                        running_selection = False  # Exit the selection loop

        # After the selection loop, replace the unselected image with a new one
        if selected_img_path is not None and running and not just_returned:
            unselected_img_index = 1 if selected_img_index == 0 else 0
            images.pop(unselected_img_index)  # Remove unselected image from the list
            
            if chosen_gender == "Male":
                new_image = random.choice(male_faces)
            elif chosen_gender == "Female":
                new_image = random.choice(female_faces)
            else:  # chosen_gender == "Both"
                new_image = random.choice(all_faces)

            images.insert(unselected_img_index, new_image)  # Insert new image into the list

        if just_returned:
            just_returned = False

        while selected_img_path is None and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    img_objs = load_images(screen, images)  
                    screen.fill((0, 0, 0))
                    draw_images(screen, img_objs)
                    draw_centered_text(screen, "Which face are you more attracted to? Use left or right arrow keys to choose", 10)
                    pygame.display.flip()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:  
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                        img_objs = load_images(screen, images)
                        screen.fill((0, 0, 0))
                        draw_images(screen, img_objs)
                        draw_centered_text(screen, "Which face are you more attracted to? Use left or right arrow keys to choose", 10)
                        pygame.display.flip()
                    
                    elif event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        selected_img_path = images[0 if event.key == pygame.K_LEFT else 1]

        if selected_img_path is not None and running:
            if selected_img_path not in chosen_images:
                chosen_images.add(selected_img_path)
                if os.path.exists(avg_image_name):
                    imgMorph = morph_faces(avg_image_name, selected_img_path, 0.5)
                    cv2.imwrite(avg_image_name, imgMorph)
                else:
                    shutil.copy2(selected_img_path, avg_image_name)

            # Remove used file names from the respective lists
            if selected_img_path in male_faces:
                male_faces.remove(selected_img_path)
            elif selected_img_path in female_faces:
                female_faces.remove(selected_img_path)
            if chosen_gender == "Both":
                if selected_img_path in all_faces:
                    all_faces.remove(selected_img_path)

            # Terminate the loop if all faces of chosen gender have been shown
            if (chosen_gender == "Male" and not male_faces) or (chosen_gender == "Female" and not female_faces) or (chosen_gender == "Both" and not all_faces):
                running = False

    showing_average = True
    while showing_average:
        screen.fill((0, 0, 0))
        avg_image = pygame.image.load("AvgAttraction.png").convert_alpha()
        screen.blit(avg_image, ((screen.get_width() - avg_image.get_width()) / 2, (screen.get_height() - avg_image.get_height()) / 2))
        draw_centered_text(screen, "Here is the average apple of your eye\n\nPress 'Q' again to quit Faces, or 'C' to continue Faces", 10)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_c:
                    images = previous_images
                    last_selected_img_path = last_selected
                    showing_average = False
                    just_returned = True
                    break


if __name__ == "__main__":
    main()