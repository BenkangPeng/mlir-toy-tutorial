#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "toy/Dialect.h"
#include "toy/Ops.h.inc"

using namespace mlir;
using namespace toy;

struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp>{
    SimplifyRedundantTranspose(MLIRContext *context):OpRewritePattern<TransposeOp>(context,1){}

    llvm::LogicalResult matchAndRewrite(TransposeOp op , PatternRewriter &rewriter) const override{
        mlir::Value transposeInput = op.getOperand();
        TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

        if(!transposeInputOp){
            return failure();
        }
        else{
            rewriter.replaceOp(op , {transposeInputOp.getOperand()});
            return success();
        }
    }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results , MLIRContext *context){
    results.add<SimplifyRedundantTranspose>(context);
}