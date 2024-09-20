module Api
  module V1
    class TransactionsController < ApplicationController
      before_action :authenticate_user!
      before_action :set_transaction, only: [:show, :update, :destroy]

      def create
        @transaction = Transaction.new(transaction_params)
        if @transaction.save
          render json: @transaction, status: :created
        else
          render json: { errors: @transaction.errors }, status: :unprocessable_entity
        end
      end

      def show
        render json: @transaction
      end

      def update
        if @transaction.update(transaction_params)
          render json: @transaction
        else
          render json: { errors: @transaction.errors }, status: :unprocessable_entity
        end
      end

      def destroy
        @transaction.destroy
        render json: { message: "Transaction deleted successfully" }, status: :ok
      end

      private

      def transaction_params
        params.require(:transaction).permit(:amount, :transaction_type, :account_id)
      end

      def set_transaction
        @transaction = Transaction.find(params[:id])
      end
    end
  end
end
