# Huggingface-T5-testing

T5 for sentence correction task of texts extracted by vision models from product images, and coorect them for semantic and lexical error, which used for downstream product attribute extraction tasks.

Used [MAVE: A Product Dataset for Multi-source Attribute Value Extraction](https://arxiv.org/abs/2112.08663) formed from the [Amazon Review Data 2018](https://nijianmo.github.io/amazon/index.html) and the [US Branded Food Product Databse](https://data.nal.usda.gov/dataset/usda-branded-food-products-database) all of which have the classified attributes for each product. Thus concatenated all these to get bulk product information and augmented them to mimic the imperfection of vision based extraction models, and then fine tuned Google's [“Unified Text-to-Text Transformer  (T5) ”](https://arxiv.org/abs/1910.10683) on this dataset to produce corrected sentences.

Trained on 4 Nvidia T4 GPUs for data parallelism with pytorch's [Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html)

More details of this and other methods on [my blogspot](https://sentencecorrectiontask.blogspot.com/)
