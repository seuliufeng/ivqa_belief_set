

SKIP_THOUGHT_DIM = 2400
SKIP_THOUGHT_WORD_DIM = 620

# try:
#     from skip_thought_util.skip_thought_model_v0 import SkipThoughtEncoder as SkipThoughtEncoder
# except:
try:
    from skip_thought_util.skip_thought_model_v1 import SkipThoughtEncoder as SkipThoughtEncoder
except:
    SkipThoughtEncoder = None

