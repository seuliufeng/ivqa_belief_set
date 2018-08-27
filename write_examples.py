from pylatex import Document, Section, Subsection, Figure, Tabular
from pylatex.basic import NewPage
from skimage.io import imread
import textwrap


def _compute_width(im_file):
    im = imread(im_file)
    try:
        h, w, _ = im.shape
    except:
        h, w = im.shape
    w = float(w) * 150. / float(h)
    return int(w)


class ExperimentWriter(object):
    def __init__(self, sv_file):
        self._sv_file = sv_file
        geo_opts = {'tmargin': '1cm', 'lmargin': '10com'}
        self._doc = Document()
        # self._doc.create(Section('Examples'))

    def add_result(self, image_id, quest_id, im_path, answer, questions):
        with self._doc.create(Section('quest_id: %d' % quest_id)):
            # add figure
            with self._doc.create(Figure(position='h!')) as fig:
                fig.add_image(im_path, width='%dpx' % _compute_width(im_path))
                fig.add_caption('Image: %d' % image_id)
            # add answer
            with self._doc.create(Subsection('%s' % answer)):
                if len(questions) % 2 == 1:
                    questions.append('')
                with self._doc.create(Tabular(table_spec='|l|l|')) as table:
                    num = int(len(questions) * 0.5)
                    table.add_hline()
                    for i in range(num):
                        a = '%s' % questions[2*i].capitalize()
                        a = textwrap.fill(a, width=45).split('\n')
                        len_a = len(a)
                        b = '%s' % questions[2*i+1].capitalize()
                        b = textwrap.fill(b, width=45).split('\n')
                        len_b = len(b)
                        max_len = max(len_a, len_b)
                        a += [''] * (max_len - len_a)
                        b += [''] * (max_len - len_b)
                        for a_i, b_i in zip(a, b):
                            table.add_row((a_i, b_i))
                        table.add_hline()
                    # table.add_hline()


                # questions
                # for q in questions:
                #     self._doc.append('\n%s' % q.capitalize())
                # with self._doc.create(Itemize()) as item:
                #     for q in questions:
                #         item.add_item(q.capitalize())
            self._doc.append(NewPage())

    def render(self):
        self._doc.generate_pdf(self._sv_file, clean=False)