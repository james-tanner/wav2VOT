from argparse import ArgumentParser
from pathlib import Path

from polyglotdb import CorpusContext
import polyglotdb.io as pgio

stops = ['B', 'D', 'G', 'P', 'T', 'K']

vowels = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
          'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
          'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
          'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'OW0', 'OW1', 'OW2',
          'OY0', 'OY1', 'OY2', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2']

def encode_types(corpus, stops_set, vowel_set, pause_labels, pause_length = 0.15):
    with CorpusContext(corpus) as c:

        if 'syllable' not in c.annotation_types:
            c.encode_type_subset('phone', stops_set, 'stops')
            c.encode_type_subset('phone', vowel_set, 'vowel')
            c.encode_syllables(syllabic_label = 'vowel')

            c.encode_pauses(pause_labels)
            c.encode_utterances(min_pause_length = pause_length)
        else:
            print("Syllabics already encoded -- skipping")
    return None

def make_query(corpus, outpath):
    with CorpusContext(corpus) as c:
        ## extract only stops
        q = c.query_graph(c.phone).filter(c.phone.subset == 'stops')

        ## surrounded by vowels
        q = q.filter(c.phone.following.subset == "vowel")
        q = q.filter(c.phone.previous.subset == "vowel")

        q = q.columns(c.phone.speaker.name.column_name('speaker'),
                      c.phone.discourse.name.column_name('discourse'),
                      c.phone.id.column_name('phone_id'),
                      c.phone.word.label.column_name('word'),
                      c.phone.syllable.label.column_name('syllable_label'),
                      c.phone.label.column_name('phone_label'),
                      c.phone.begin.column_name('phone_begin'),
                      c.phone.end.column_name('phone_end')
        )
    q.to_csv(outpath)
    return None

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("corpus_dir", type = Path)
    parser.add_argument("csv_path", type = Path)
    parser.add_argument("--reload", action = "store_true")
    args = parser.parse_args()

    corpus = args.corpus_dir
    corpus_name = corpus.stem

    if args.reload:
        mfa = pgio.inspect_mfa(corpus)
        mfa.call_back = print
        with CorpusContext(corpus_name) as c:
            try:
                print("Resetting corpus")
                c.reset()
            except Exception as e:
                print(e)
                pass
            print("Loading corpus")
            c.load(mfa, corpus)

            print("Encoding syllables")
            encode_types(corpus_name, stops, vowels, "<SIL>")

    print("Writing CSV")
    make_query(corpus_name, str(args.csv_path))
    
