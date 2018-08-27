from models.model_creater import get_model_creation_fn


def test():
    from config import ModelConfig
    model_fn = get_model_creation_fn('VAQ-CA')
    config = ModelConfig()
    config.top_answer_file = 'data/top_answer2000_sequences.h5'
    config.vqa_agent_ckpt = '/import/vision-ephemeral/fl302/code/vqa2.0/model/' \
                            'curr_VQA-Soft_Res5c/model.ckpt-325000'
    model = model_fn(config, phase='train')
    model.build()


if __name__ == '__main__':
    test()

