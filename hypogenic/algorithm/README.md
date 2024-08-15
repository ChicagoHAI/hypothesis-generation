### Algorithm
The "algorithm" folder is the center of mass of hypogenic - the entire repo revolves around it.  

The three folders - generation, inference, and update - form hypogenic's skeleton.  Each folder contains some core functionality that will be explained later.

Each of these components are modularized by a base class (found in the respective folder's base.py) and a default implementation (default.py).
This structure allows us to implement our own versions of hypogenic by rewriting the base class, or we can simply opt for the default implementation.

The class' use cases are as follows:

**Generation**
We use the generation class to initalize hypotheses either at the start of generation, or once we surpass the reward threshold and wish to make new hypotheses.  It supports batching the generation of multiple new hypotheses.

**Inference**
The inference class is used in both the *Generation* and *Update* classes to track your hypothesis' accuracy.  It supports batching as well.

**Update**
Update is the main hypogenic loop where we wait until our reget exceeds a certain capacity by introducing new samples; once the threshold is met, we generate new hypotheses using the examples that our hypothesis bank performed poorly on.

