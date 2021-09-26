from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts



def main():
    opts = get_opts()

    # ## Q1.1
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255 # normalize
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # print('Q1.1 filter_responses', filter_responses.shape)
    # util.display_filter_responses(opts, filter_responses)

    # # ## Q1.2
    # n_cpu = util.get_num_CPU()
    # visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    # # # # # Q1.3
    # # name = 'kitchen/sun_aasmevtpkslccptd.jpg'
    # # name = 'aquarium/sun_azipgwtixazwwduo.jpg'
    name = 'park/labelme_djdpufqmxpltzcp.jpg'
    img_path = join(opts.data_dir, name)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(name, img, wordmap)

    # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, wordmap, n_worker=n_cpu)

    ## Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
