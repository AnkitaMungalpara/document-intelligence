# Document Parsing with Nougat: Transforming Scientific PDFs into Machine-Readable Text

## Introduction

Scientific knowledge is predominantly stored in PDF documents, which present a significant challenge for information extraction and processing. While PDFs are human-readable, they lack the semantic structure needed for machine processing, especially when dealing with complex elements like mathematical equations, tables, and figures. This is where Nougat comes in.

Nougat (Neural Optical Understanding for Academic Documents) is a state-of-the-art model developed by Meta AI that transforms scientific PDF documents into structured, machine-readable text. Unlike traditional OCR systems that process text line by line, Nougat understands the relationships between different elements on a page, making it particularly effective for scientific documents with complex layouts and mathematical expressions.

In this blog post, we'll explore how to use Nougat to parse scientific documents and extract structured text, including mathematical formulas, from PDFs.

## Understanding Nougat's Architecture

Nougat is built as a Visual Transformer model that performs an Optical Character Recognition (OCR) task specifically designed for scientific documents. The model architecture consists of:

1. **Vision Encoder**: Processes the input image (PDF page) and extracts visual features
2. **Decoder**: Converts these visual features into structured text in a markup language

The model uses a Vision Encoder-Decoder architecture from Hugging Face's Transformers library, specifically the `VisionEncoderDecoderModel` with the "facebook/nougat-base" pre-trained weights.

## Setting Up the Environment

To get started with Nougat, we need to install the necessary dependencies:

```python
# Install required packages
!pip install -q pymupdf python-Levenshtein nltk
!pip install -q git+https://github.com/huggingface/transformers.git
```

Next, we import the required libraries:

```python
from transformers import NougatProcessor, VisionEncoderDecoderModel, infer_device
from datasets import load_dataset
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from typing import Optional, List
import io
import fitz
from pathlib import Path
```

## Loading the Nougat Model

We load the pre-trained Nougat model and processor:

```python
# Load the processor and model
processor = NougatProcessor.from_pretrained("facebook/nougat-base")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# Move model to appropriate device (GPU if available)
device = infer_device()
model.to(device)
```

## Processing PDFs with Nougat

### Step 1: Rasterizing PDF Pages

Before we can process a PDF with Nougat, we need to convert each page into an image. We'll use PyMuPDF (fitz) for this:

```python
def rasterize_paper(
    pdf: Path,
    outpath: Optional[Path] = None,
    dpi: int = 96,
    return_pil=False,
    pages=None,
) -> Optional[List[io.BytesIO]]:
    """
    Rasterize a PDF file to PNG images.
    
    Args:
        pdf (Path): The path to the PDF file.
        outpath (Optional[Path]): Output directory for images.
        dpi (int): The output DPI. Defaults to 96.
        return_pil (bool): Return PIL images instead of saving to disk.
        pages (Optional[List[int]]): Pages to rasterize. If None, all pages.
        
    Returns:
        Optional[List[io.BytesIO]]: PIL images if return_pil is True.
    """
    pillow_images = []
    if outpath is None:
        return_pil = True
    try:
        if isinstance(pdf, (str, Path)):
            pdf = fitz.open(pdf)
        if pages is None:
            pages = range(len(pdf))
        for i in pages:
            page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
            if return_pil:
                pillow_images.append(io.BytesIO(page_bytes))
            else:
                with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                    f.write(page_bytes)
    except Exception:
        pass
    if return_pil:
        return pillow_images
```

### Step 2: Implementing Custom Stopping Criteria

Nougat uses a custom stopping criteria during text generation to avoid repetitive outputs:

```python
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict

class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)


class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0
```

This stopping criteria monitors the variance in the model's output logits and stops generation when it detects repetitive patterns, which is crucial for handling long scientific documents.

### Step 3: Creating a Transcription Function

Let's create a function that combines all the steps to transcribe a PDF page:

```python
def transcribe_image(image, processor, model, device):
    """
    Transcribes a single image using the Nougat model.

    Args:
        image (PIL.Image.Image): The input image.
        processor: The Nougat processor.
        model: The Nougat model.
        device: The device to run the model on.

    Returns:
        str: The transcribed text.
    """
    pixel_values = processor(image, return_tensors="pt").pixel_values
    outputs = model.generate(
        pixel_values.to(device),
        min_length=1,
        max_length=4500,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
    )
    generated_sequence = processor.batch_decode(
        outputs[0],
        skip_special_tokens=True,
    )[0]
    generated_sequence = processor.post_process_generation(
        generated_sequence,
        fix_markdown=False
    )
    return generated_sequence
```

## Nougat in Action: Parsing a Scientific Paper

Let's see Nougat in action by processing pages from a scientific paper:

```python
# Download a sample PDF
filepath = hf_hub_download(repo_id="ysharma/nougat", filename="input/nougat.pdf", repo_type="space")

# Convert PDF pages to images
images = rasterize_paper(pdf=filepath, return_pil=True)

# Process the first page
image = Image.open(images[0])
transcribed_text = transcribe_image(image, processor, model, device)
print(transcribed_text)
```

The output from the first page reveals the power of Nougat:

```
# Nougat: Neural Optical Understanding for Academic Documents

 Lukas Blecher

Correspondence to: lblecher@meta.com

Guillem Cucurull

Thomas Scialom

Robert Stojnic

Meta AI

###### Abstract

Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (**N**eural **O**ptical **U**nderstanding for **A**cademic Documents), a Visual Transformer model that performs an _Optical Character Recognition_ (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.
```

## Handling Mathematical Expressions

One of Nougat's most impressive capabilities is its ability to accurately transcribe mathematical expressions. Let's look at a page containing complex equations:

```python
# Process a page with mathematical content
image = Image.open(images[8])
math_text = transcribe_image(image, processor, model, device)
print(math_text)
```

Output:

```
Here \(\ell\) is the signal of logits and \(x\) the index. Using this new signal we compute variances again but this time from the point \(x\) to the end of the sequence

\[\mathrm{VarEnd}_{B}[\bm{\ell}](x)=\frac{1}{S-x}\sum_{i=x}^{S}\left(\mathrm{Var Win}_{B}[\bm{\ell}](i)-\frac{1}{S-x}\sum_{j=x}^{S} \mathrm{VarWin}_{B}[\bm{\ell}](i)\right)^{2}.\]

If this signal drops below a certain threshold (we choose 6.75) and stays below for the remainder of the sequence, we classify the sequence to have repetitions.
```

As you can see, Nougat correctly transcribes the complex mathematical expressions using LaTeX notation, making them machine-readable while preserving their semantic meaning.

## Challenges and Limitations

While Nougat represents a significant advancement in document parsing, it does have some limitations:

1. **Language Limitations**: The model is primarily trained on English documents, though it can handle other Latin-based languages to some extent.

2. **Mathematical Expression Ambiguity**: LaTeX offers multiple ways to express the same mathematical formula, leading to potential discrepancies between prediction and ground truth.

3. **Boundary Detection**: It can be challenging to determine where inline math environments end and regular text begins, affecting both math and plain text scores.

4. **Document Structure**: Nougat works best with research papers and documents with similar structures, though it can still process other types of documents.

As noted in the paper: "The expected score for mathematical expressions is lower than for plain text" due to these inherent ambiguities.

## Applications

Nougat opens up numerous possibilities for scientific document processing:

1. **Knowledge Extraction**: Extract structured information from scientific papers for analysis and database building.

2. **Accessibility**: Convert PDF-based scientific content into accessible formats for screen readers and other assistive technologies.

3. **Search and Indexing**: Enable more effective search across scientific literature by converting mathematical expressions into searchable text.

4. **Data Mining**: Facilitate large-scale analysis of scientific literature by converting PDFs into machine-readable formats.

## Conclusion

Nougat represents a significant advancement in the field of document understanding, particularly for scientific papers with complex layouts and mathematical expressions. By bridging the gap between human-readable PDFs and machine-readable text, it enhances the accessibility and usability of scientific knowledge in the digital age.

The model's ability to accurately transcribe not just plain text but also complex mathematical expressions and tables makes it an invaluable tool for researchers, publishers, and anyone working with scientific literature. As the technology continues to evolve, we can expect even more accurate and comprehensive document parsing capabilities.

## References

1. [Nougat: Neural Optical Understanding for Academic Documents - Meta AI](https://arxiv.org/pdf/2308.13418)
2. [Hugging Face Transformers Nougat Model](https://huggingface.co/facebook/nougat-base)
