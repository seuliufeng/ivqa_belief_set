from readers.vert_base_data_fetcher import AttentionDataPrefetcher





if __name__ == '__main__':
    from time import time
    # from inference_utils.question_generator_util import SentenceGenerator

    # to_sentence = SentenceGenerator(trainset='trainval')
    reader = AttentionDataPrefetcher(None, 1, batch_size=4, subset='kpval', sample_negative=True)
    for i in range(10):
        print(i)
        data = reader.pop_batch()
        import pdb
        pdb.set_trace()