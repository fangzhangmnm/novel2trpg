import re
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple


import re

import re

import re

def convert_txt_to_md(txt_path, md_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Step 1: remove heading blank space for each line using re
    lines = [re.sub(r'^\s+', '', line) for line in lines]

    
    
    # Step 2: separate characters using re_chapter_title
    re_chapter_title = re.compile(r'^(第.{1,9}(?=章|卷|篇|集|话)|序章|序|后记|番外篇)', flags=re.MULTILINE)
    chapters = re_chapter_title.split(''.join(lines))[1:]
    
    chapters=[chapters[i]+chapters[i+1] for i in range(0,len(chapters),2)]

    # # replace [] with 「」 for non links
    chapters = [chapter.replace('[', '「').replace(']', '」') for chapter in chapters]


    # Step 3: merge short<200 char or blank chapters to the next chapter
    new_chapters = [chapters[0]]
    for i in range(1, len(chapters)):
        if len(new_chapters[-1]) < 200 or new_chapters[-1].strip() == '':
            new_chapters[-1] += chapters[i]
        else:
            new_chapters.append(chapters[i])
    
    # Step 4: convert to md
    with open(md_path, 'w', encoding='utf-8') as f:
        for chapter in new_chapters:
            lines=chapter.split('\n')
            if len(lines)>1:
                f.write(f'# {lines[0]}\n')
                f.write('\n\n'.join(lines[1:]))
    
    print(f'Converted {txt_path} to {md_path}')


def read_novel(novel_path)->Dict[int,Dict]:
    with open(novel_path) as f:
        all_text = f.read()

    all_text=re.sub(r'\r', '', all_text)
    all_text=re.sub(r'\n+', '\n', all_text)
    all_text=re.sub(r'^\s+', '', all_text,flags=re.MULTILINE)

    re_chapter_title=re.compile(r'^(# .+)', flags=re.MULTILINE)
    splits=re.split(re_chapter_title,all_text)[1:]
    titles=splits[::2]
    texts=splits[1::2]
    chapters={chapter_id:{'title':title,'text':text,'chapter_id':chapter_id} for chapter_id,(title,text) in enumerate(zip(titles,texts))}
    return chapters

from langchain.schema import Document
import re

from langchain.schema import Document
import re

def read_wiki(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    wiki_documents = []
    current_title = []
    current_content = []

    for line in lines:
        match = re.match(r'^(#+)\s+(.*)', line)
        if match:
            level = len(match.group(1))
            title = match.group(2)
            current_title[level-1:] = [title]
            current_content.append(line[len(match.group(0)):])
        else:
            current_content.append(line)

        if not line.strip() or line == lines[-1]:
            content = ''.join(current_content).strip()
            if content:
                metadata = {'title': ' '.join(current_title)}
                text = metadata['title'] + ' ' + content.replace('\n', ' ')
                wiki_documents.append(Document(metadata=metadata, page_content=text))
            current_content = []

    return wiki_documents



def save_chapter_titles(chapters, chapter_titles_path):
    # save the chapter titles into a txt file, with chapter length and list of chunk_ids
    with open(chapter_titles_path, 'w') as f:
        for chapter in chapters.values():
            f.write(f"{chapter['chapter_id']} {chapter['title']}\t{len(chapter['text'])}\t{chapter.get('chunk_ids', [])}\n")
    print(f"Saved chapter titles to {chapter_titles_path}")


def split_chapters(chapters, chunk_len)->Tuple[Dict[int,Dict],Dict[int,Dict]]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_len, chunk_overlap=0)
    documents=[Document(metadata={'title':chapter['title'],'chapter_id':chapter['chapter_id']},page_content=chapter['text']) for chapter in chapters.values()]
    
    documents = text_splitter.split_documents(documents)
    # save chunk_id into chapters
    chunks = {}
    for iDocument, document in enumerate(documents):
        chunks[iDocument] = {
            'title': document.metadata['title'],
            'text': document.page_content,
            'chunk_id': iDocument,
            'chapter_id': document.metadata['chapter_id'],
        }
        if iDocument>0 and chunks[iDocument-1]['chapter_id']==chunks[iDocument]['chapter_id']:
            chunks[iDocument]['previous_chunk_id']=iDocument-1
            chunks[iDocument-1]['next_chunk_id']=iDocument
        chapters[document.metadata['chapter_id']].setdefault('chunk_ids', []).append(iDocument)

        
    return chapters, chunks

def split_subchunk_(chunk,sub_chunk_len)->Dict:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=sub_chunk_len, chunk_overlap=0)
    documents=[Document(metadata={'title':chunk['title'],'chunk_id':chunk['chunk_id']},page_content=chunk['text'])]
    documents = text_splitter.split_documents(documents)
    chunk['sub_chunks']=[{'title':document.metadata['title'],'text':document.page_content,'sub_chunk_id':iSubChunk} for iSubChunk,document in enumerate(documents)]
    return chunk


def load_chunks(novel_chunk_dir)->Dict[int,Dict]:
    chunks = dict()
    for filename in os.listdir(novel_chunk_dir):
        if filename.startswith('CHUNK_'):
            with open(os.path.join(novel_chunk_dir, filename), 'r') as f:
                chunk = json.load(f)
            chunks[chunk['chunk_id']] = chunk
    return chunks

def save_chunks(chunks, novel_chunk_dir):
    os.makedirs(novel_chunk_dir, exist_ok=True)
    for filename in os.listdir(novel_chunk_dir):
        if filename.startswith('CHUNK_'):
            chunk_id = int(filename.split('_')[1].split('.')[0])
            if chunk_id not in chunks.keys():
                os.remove(os.path.join(novel_chunk_dir, filename))
                print('Removed', os.path.join(novel_chunk_dir, filename))
    nSaved = 0
    for chunk_id, chunk in chunks.items():
        chunk_path = os.path.join(novel_chunk_dir, 'CHUNK_{0:05d}.json'.format(chunk_id))
        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        nSaved += 1
    print('Saved {0} chunks to'.format(nSaved), novel_chunk_dir)

def save_chunk(chunk, novel_chunk_dir):
    chunk_path = os.path.join(novel_chunk_dir, 'CHUNK_{0:05d}.json'.format(chunk['chunk_id']))
    with open(chunk_path, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    print('Saved chunk to', chunk_path)

def load_chapters(novel_chapter_dir)->Dict[int,Dict]:
    chapters = dict()
    for filename in os.listdir(novel_chapter_dir):
        if filename.startswith('CHAPTER_'):
            with open(os.path.join(novel_chapter_dir, filename), 'r') as f:
                chapter = json.load(f)
            chapters[chapter['chapter_id']] = chapter
    return chapters

def save_chapters(chapters, novel_chapter_dir):
    os.makedirs(novel_chapter_dir, exist_ok=True)
    for filename in os.listdir(novel_chapter_dir):
        if filename.startswith('CHAPTER_'):
            chapter_id = int(filename.split('_')[1].split('.')[0])
            if chapter_id not in chapters.keys():
                os.remove(os.path.join(novel_chapter_dir, filename))
                print('Removed', os.path.join(novel_chapter_dir, filename))
    nSaved = 0
    for chapter_id, chapter in chapters.items():
        chapter_path = os.path.join(novel_chapter_dir, 'CHAPTER_{0:05d}.json'.format(chapter_id))
        with open(chapter_path, 'w', encoding='utf-8') as f:
            json.dump(chapter, f, ensure_ascii=False, indent=2)
        nSaved += 1
    print('Saved {0} chapters to'.format(nSaved), novel_chapter_dir)

def save_chapter(chapter, novel_chapter_dir):
    chapter_path = os.path.join(novel_chapter_dir, 'CHAPTER_{0:05d}.json'.format(chapter['chapter_id']))
    with open(chapter_path, 'w', encoding='utf-8') as f:
        json.dump(chapter, f, ensure_ascii=False, indent=2)
    print('Saved chapter to', chapter_path)


def plot_chapter_length(chapters):
    plt.figure(figsize=(5, 2))
    plt.hist([len(chapter['text']) for chapter in chapters.values()], bins=np.logspace(0, 10, 100))
    plt.xscale('log')
    plt.xlabel('Chapter length')
    plt.ylabel('Count')
    plt.title('Distribution of chapter length')
    plt.show()
    print('Number of chapters: ', len(chapters))


