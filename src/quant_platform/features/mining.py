"""Bounded AST interpreter. No eval, attributes, file/network/process capabilities."""
import ast
import hashlib
import time
import numpy as np
import pandas as pd
from quant_platform.research.governance import Ledger

NAMES={'open','high','low','close','volume'}
FUNCTIONS={'lag','mean','std','abs','rank'}
class FactorDSL:
    def parse(self,expression):
        if len(expression)>2000:raise ValueError('expression too long')
        tree=ast.parse(expression,mode='eval');nodes=list(ast.walk(tree))
        allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd)
        if len(nodes)>100 or any(not isinstance(n,allowed) for n in nodes):raise ValueError('AST capability denied')
        for node in nodes:
            if isinstance(node,ast.Name) and node.id not in NAMES|FUNCTIONS:raise ValueError('name denied')
            if isinstance(node,ast.Constant) and (isinstance(node.value,bool) or not isinstance(node.value,(int,float)) or not np.isfinite(node.value) or abs(node.value)>1e6):raise ValueError('constant denied')
            if isinstance(node,ast.Call):
                if not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS or node.keywords:raise ValueError('call denied')
                if node.func.id in {'lag','mean','std','rank'}:
                    if len(node.args)!=2 or not isinstance(node.args[1],ast.Constant) or type(node.args[1].value) is not int or not 1<=node.args[1].value<=365:raise ValueError('past-only bounded window required')
                elif len(node.args)!=1:raise ValueError('arity')
        return tree
    def evaluate(self,expression,frame):
        tree=self.parse(expression)
        if len(frame)>100000:raise ValueError('row budget exceeded')
        deadline=time.monotonic()+2
        def visit(n,depth=0):
            if depth>15 or time.monotonic()>deadline:raise TimeoutError('DSL budget exceeded')
            if isinstance(n,ast.Expression):return visit(n.body,depth+1)
            if isinstance(n,ast.Constant):return n.value
            if isinstance(n,ast.Name):return frame[n.id]
            if isinstance(n,ast.UnaryOp):return -visit(n.operand,depth+1) if isinstance(n.op,ast.USub) else visit(n.operand,depth+1)
            if isinstance(n,ast.BinOp):
                a,b=visit(n.left,depth+1),visit(n.right,depth+1)
                if isinstance(n.op,ast.Add):return a+b
                if isinstance(n.op,ast.Sub):return a-b
                if isinstance(n.op,ast.Mult):return a*b
                with np.errstate(divide='ignore',invalid='ignore'):return a/b
            if isinstance(n,ast.Call):
                a=visit(n.args[0],depth+1);name=n.func.id
                if name=='abs':return abs(a)
                window=n.args[1].value
                if name=='lag':return a.shift(window)
                if name=='mean':return a.rolling(window).mean()
                if name=='std':return a.rolling(window).std()
                if name=='rank':return a.rolling(window).rank(pct=True)
            raise ValueError('unhandled AST')
        result=visit(tree)
        if not isinstance(result,pd.Series):result=pd.Series(result,index=frame.index)
        return result.replace([np.inf,-np.inf],np.nan)
    def record_candidate(self,expression,frame,path):
        tree=self.parse(expression);canonical=ast.dump(tree,include_attributes=False);candidate=hashlib.sha256(canonical.encode()).hexdigest()
        ledger=Ledger(path)
        if any(r['record'].get('candidate_id')==candidate for r in ledger.rows()):return {'candidate_id':candidate,'duplicate':True}
        value=self.evaluate(expression,frame)
        ledger.append(dict(candidate_id=candidate,expression=expression,canonical_ast=canonical,status='PENDING_HUMAN_APPROVAL',
                           finite_rows=int(value.notna().sum()),auto_mining_enabled=False,multiple_testing_trial=True))
        return {'candidate_id':candidate,'duplicate':False,'status':'PENDING_HUMAN_APPROVAL'}
