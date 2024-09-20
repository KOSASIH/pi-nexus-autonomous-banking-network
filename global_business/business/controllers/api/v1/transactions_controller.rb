class Api::V1::TransactionsController < ApplicationController
  before_action :authenticate_user!

  def index
    transactions = current_user.accounts.find(params[:account_id]).transactions
    render json: transactions, status: :ok
  end

  def show
    transaction = current_user.accounts.find(params[:account_id]).transactions.find(params[:id])
    render json: transaction, status: :ok
  end

  def create
    transaction = current_user.accounts.find(params[:account_id]).transactions.new(transaction_params)
    if transaction.save
      render json: transaction, status: :created
    else
      render json: { errors: transaction.errors }, status: :unprocessable_entity
    end
  end

  def update
    transaction = current_user.accounts.find(params[:account_id]).transactions.find(params[:id])
    if transaction.update(transaction_params)
      render json: transaction, status: :ok
    else
      render json: { errors: transaction.errors }, status: :unprocessable_entity
    end
  end

  def destroy
    transaction = current_user.accounts.find(params[:account_id]).transactions.find(params[:id])
    transaction.destroy
    render json: {}, status: :no_content
  end

  private

  def transaction_params
    params.require(:transaction).permit(:amount, :type)
  end
end
