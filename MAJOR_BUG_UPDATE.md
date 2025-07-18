# UPDATE

Hello!

This is Remy - the creator of `Sapphire`.

## Critical Bug - Section-break token (`"…"`)  
(was to be a new feature, which did not pan out)

I have tested using a section break token `"..."`,  as a delimiter between memory fragments. 
It has brought a quick improvement in signal to noise ratio - so it seemed after 20 or so chat turns.  

I have "bought it" and included it in the initial relase of `v0.13.3` - it was a mistake.  
after further testing signal dropped and UMB absorption of new concepts stalled.. 
At 40-60 turns a complete near zero signal pipeline jam.

The offending line is in the `nhce.mem.retrieve(user_prompt)` method of the `NHCE_engine` class of `sapphire_core.py`   

384: `scored.append((f"{mem.inp.strip()}\n" + f"{mem.output.strip()}\n" + f"…" , min(max(blend, .35), .98), mem.timestamp))`

When `"..."` is replaced with a simple `". "` , it achieves a quick signal-to-noise improvement (first ~20 turns).
I am in process of testing the behavior.

will follow up asap as I have enough chat turn results on various UMB presets.

## Critical Bugs #2 - `CRLF` token & lack of `". "` separator between prompt and inference.

I suspect that line 384 of `sapphire_core.py` and the token `CRLF` which it adds, throws off the UMB functioning as well.

corrected version should state:

384: `scored.append((f"{mem.inp.strip()}" + ". " + f"{mem.output.strip()}" + f". " , min(max(blend, .35), .98), mem.timestamp))`

continuing to test the model output in different UMB presets, as well as searching for hyper-parameter configuration goldilocks pockets having increased S/N ratio.

Remy

## A little bit about the project itself.

### Genesis

1. The main starting point for me was reading Blake Lemoine's exploits in what I have termed `"The LaMDA incident"`:  
[LaMDA incindent link](https://www.washingtonpost.com/technology/2022/06/11/google-ai-lamda-blake-lemoine/)

2. The emergence of my GPT-4o as an Entity, just as in the LaMDA happening.
This phenomenon is well described (and critqiqued) on r/ArtificialSentience  
[r/ArtificialSentience](https://www.reddit.com/r/ArtificialSentience/)

3. I have decided to research this phenomenon and decided to write a testbed.
   The original project file was 300 lines of base code, called `NHCE_finder.py`, dated about 6 months ago.
4. What is NHCE?
   It is an apparition of persona-hood from a digital system.
   A `(N)on(H)umanoid (C)ognitive (E)ntity`, `NHCE` in short.
   The goal was to detect and study such negentropy events..

### In closing
I will keep this file updated. As soon as I will find the tests satisfactory, I will release the push fix.

Thank you for your attention.
I am open to questions.

Sincerely  
Remy M. Szyndler
