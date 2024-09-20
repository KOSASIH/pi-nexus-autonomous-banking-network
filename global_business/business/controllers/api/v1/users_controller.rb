module Api
  module V1
    class UsersController < ApplicationController
      before_action :authenticate_user!, except: [:create]
      before_action :set_user, only: [:show, :update, :destroy]

      def create
        @user = User.new(user_params)
        if @user.save
          render json: @user, status: :created
        else
          render json: { errors: @user.errors }, status: :unprocessable_entity
        end
      end

      def show
        render json: @user
      end

      def update
        if @user.update(user_params)
          render json: @user
        else
          render json: { errors: @user.errors }, status: :unprocessable_entity
        end
      end

      def destroy
        @user.destroy
        render json: { message: "User deleted successfully" }, status: :ok
      end

      private

      def user_params
        params.require(:user).permit(:username, :email, :password, :first_name, :last_name, :role)
      end

      def set_user
        @user = User.find(params[:id])
      end
    end
  end
end
