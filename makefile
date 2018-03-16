run:
	python emotion_task.py

clean:
	rm -vf corpus/debug/processing_test.txt
	rm -vf corpus/debug/processing_train.txt
	rm -vf corpus/debug/vector_test.txt
	rm -vf corpus/debug/vector_train.txt
	rm -vf corpus/debug/word_list.txt
	rm -vf corpus/debug/emb_train.txt
	rm -vf corpus/debug/model
