# Refactoring principles

General rules derived from Robert C. Martin's *Clean Code*, Chapters 2, 3,
and 6. Guidance about mathematical notation and numerical behavior adapts
these principles to mathematical software.

## Chapter 2 — Meaningful Names

1. **Reveal intent and preserve meaning.** A name should identify what a
   quantity represents or what an operation accomplishes. Understand the
   domain before renaming. Distinguish genuinely different concepts, and
   avoid names that imply a different quantity, role, or representation.

2. **Use consistent domain vocabulary.** Prefer established technical terms
   to invented descriptions. Use one term for one concept throughout related
   code; use different terms for different concepts. Names should help
   readers connect the implementation to its domain, not require translation
   through a private naming scheme.

3. **Provide enough context, not maximum verbosity.** Choose names that are
   clear within their scope and searchable where they are widely used.
   In mathematical code, conventional symbols are appropriate when their
   meanings are defined nearby. Expand a name when it clarifies intent;
   spelling out a formula in words is not inherently clearer. Avoid redundant
   prefixes, type encodings, and context already supplied by the enclosing code.

## Chapter 3 — Functions

1. **Give each function one conceptual responsibility.** Keep functions
   small enough that their purpose and structure are readily understood.
   Separate unrelated tasks rather than selecting them with behavior flags.
   For mathematical software, cases of one parameterized operator can belong
   to the same responsibility; branching alone does not imply multiple jobs.

2. **Keep each function at one level of abstraction.** Describe a high-level
   operation through meaningful steps, with detailed calculations inside
   those steps. A reader should not have to alternate between understanding
   the overall algorithm and decoding low-level mechanics in the same body.

3. **Arrange code in descending explanatory order.** Follow the Stepdown
   Rule: introduce an operation before the helpers that explain its next
   level of detail. Apply this reading order across the file, grouping
   related operations together. Short section labels can reveal the
   structure, but should describe real responsibilities rather than invent
   a hierarchy among peer operations.

4. **Extract meaningful concepts and remove duplicated knowledge.** A helper
   should name a recognizable operation, not merely restate a block's code.
   Centralize a calculation or rule when callers genuinely share it. Keep
   tightly connected mathematical reasoning together, and avoid fragmenting
   it into helpers that add navigation without explanatory value.

5. **Make inputs, results, and effects explicit.** Minimize the information
   callers must supply and remember. Group arguments when they form a real
   domain concept, not merely to shorten a parameter list. Prefer returned
   results to disguised output arguments. Separate queries from commands,
   expose necessary ordering dependencies, and make mutation apparent.
   For mathematical operations, state assumptions and output guarantees so
   callers need not infer them from the implementation.

6. **Refine structure while preserving behavior.** Use tests to support
   incremental renaming, extraction, deduplication, and reordering. Treat
   changes to the supported behavior as separate decisions. In numerical
   software, algebraic equivalence alone does not establish equivalence of
   floating-point results, boundary conventions, or safeguards.

7. update states rather than create many returns

8. Prefer execption to returning error code.

9. structured programming

## Chapter 6 — Objects and Data Structures

1. **Abstract meaning, not merely storage.** An object should expose useful
    operations while hiding how its data is represented. Private fields with
    matching getters and setters may still expose the entire representation.
    Judge an abstraction by what callers can accomplish without knowing its
    internals, not by whether the code is wrapped in a class.

2. **Choose the representation according to the expected kind of change.**
    Objects with a common behavioral interface make new types easier to add,
    while new operations may require changing every type. Data structures
    with separate functions make new operations easier to add, while new
    data types may require changing every function that handles them.
    Neither approach is universally superior.

3. **Keep objects and data structures conceptually distinct.** Plain data
    structures may expose their contents for other functions to process.
    Objects should own their behavior and protect their representation and
    invariants. Avoid hybrids that offer encapsulated behavior while also
    requiring callers to manipulate the same internal state directly.

4. **Depend on behavior rather than internal navigation.** Ask an object
    for the operation needed instead of traversing its internal object graph
    to perform the operation elsewhere. This is the intent of the Law of
    Demeter, not a mechanical restriction on the number of dots in an
    expression. Reading openly exposed data structures is a different case;
    introducing accessors alone does not turn them into abstractions.


## Chapter 8 - Boundaries.
how to manage the third-party dependency


## Chapter 10 - Clean Classes
- there is seldom a good reason to have a public variable
- step-down rules
- class should be small
