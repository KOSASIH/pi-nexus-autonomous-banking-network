(defpackage :agi
  (:use :cl)
  (:export #:agi))

(in-package :agi)

(defstruct agi
  (knowledge-base nil :type list)
  (inference-engine nil :type function))

(defun create-agi (knowledge-base inference-engine)
  (make-agi :knowledge-base knowledge-base :inference-engine inference-engine))

(defun reason (agi input)
  (funcall (agi-inference-engine agi) (agi-knowledge-base agi) input))

(defun learn (agi input output)
  (push (cons input output) (agi-knowledge-base agi)))

# Example usage
(defparameter *agi* (create-agi nil #'inference-engine))

(defun inference-engine (knowledge-base input)
  (cond ((assoc input knowledge-base) => cdr)
        (t (error "Unknown input"))))

(learn *agi* "What is the capital of France?" "Paris")
(learn *agi* "What is the capital of Germany?" "Berlin")

(print (reason *agi* "What is the capital of France?")) ; Output: "Paris"
(print (reason *agi* "What is the capital of Germany?")) ; Output: "Berlin"
