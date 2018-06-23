;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; air cargo domain from AIMA book 2nd ed.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; since the 'at' predicate is used for both cargo and planes, I didn't specify types
; in this domain to keep things simpler.

(define (domain aircargo)
  (:requirements :strips)
  (:predicates (at ?x ?a)
	       (cargo ?c)
	       (airport ?a)
	       (plane ?p)
	       (in ?x ?p)
	       )

  (:action load
	     :parameters (?c ?p ?a)
	     :precondition (and (cargo ?c) (plane ?p) (airport ?a) (at ?c ?a) (at ?p ?a))
	     :effect  (and (in ?c ?p) (not (at ?c ?a))))

  (:action unload
	     :parameters (?c ?p ?a)
    	 :precondition (and (cargo ?c) (plane ?p) (airport ?a) (in ?c ?p) (at ?p ?a))
	     :effect (and (at ?c ?a) (not (in ?c ?p))))

  (:action fly
	     :parameters (?p ?f ?t)
    	 :precondition (and (at ?p ?f) (plane ?p) (airport ?f) (airport ?t))
	     :effect (and (at ?p ?t) (not (at ?p ?f))))
)