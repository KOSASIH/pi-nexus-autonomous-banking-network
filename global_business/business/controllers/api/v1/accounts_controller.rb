module Api
  module V1
    class AccountsController < ApplicationController
      before_action :authenticate_user!
      before_action :set_account, only: [:show, :update, :destroy]

      def create
        @account = Account.new(account_params)
        if @account.save
          render json: @account, status: :created
        else
          render json: { errors: @account.errors }, status: :unprocessable_entity
        end
      end

      def show
        render json: @account
      end

      def update
        if @account.update(account_params)
          render json: @account
        else
          render json: { errors: @account.errors }, status: :unprocessable_entity
        end
      end

      def destroy
        @account.destroy
        render json: { message: "Account deleted successfully" }, status: :ok
      end

      private

      def account_params
        params.require(:account).permit(:account_number, :routing_number, :account_type, :user_id)
      end

      def set_account
        @account = Account.find(params[:id])
      end
    end
  end
end
