import config
import facenet
import os

if __name__ == "__main__":
    # load parameter
    args_margin = config.args_margin
    args_image_size = config.args_image_size

    minsize = config.minsize  # minimum size of face
    threshold = config.threshold  # three steps's threshold
    factor = config.factor  # scale factor
    face_threshold = config.face_threshold

    db_name = config.db_name
    known_img_path = config.known_img_path
    update = config.update
    facenet_model = config.args_model

    # update database
    db_face_vectors, db_face_source, _ = facenet.faceDB(db_name, img_path=known_img_path, update=update)

    # show database
    output_json = list()

    for each_source in range(len(db_face_source)):
        name = db_face_source[each_source].split('/')[-2]
        vector = db_face_vectors[each_source]
        output_json.append({'face_name': name, "face_feature": vector})

    for each in output_json:
        print("person name: %s" % each['face_name'])
        # print("face feature: ", each['face_feature'])

